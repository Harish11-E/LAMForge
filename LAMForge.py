"""
LAMForge
Automatic topology and force-field generator for LAMMPS

Author: Konstantinos Xanthopoulos
License: MIT
"""

import os
import math
import itertools
import traceback
import json
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import Counter

from pymatgen.core import Structure, Element
import importlib.util
import numpy as np  # for QEq linear algebra

# -------------------- Force field parameter loader --------------------


class UFFParameters:
    """
    Wrapper around force-field parameters, especially UFF.
    Supports:
      1) Python file with a dict named UFF_DATA:
         UFF_DATA = {
             "Zn3+2": (r1, theta0, x1, D1, zeta, Z1, Vi, Uj, Xi, Hard, Radius),
             ...
         }
      2) Legacy text format (one type per line):
         type bond angle r_star epsilon scale q_eff
    We store:
      bond   -> r1 (Å, UFF "bond radius")
      angle  -> theta0 (deg, valence angle)
      r_star -> x1 (Å, vdW minimum position)
      epsilon-> D1 (kcal/mol)
      Z      -> Z1 (effective charge)
      chi    -> Xi (UFF electronegativity)
      hard   -> Hard (UFF hardness, used for QEq)
      radius -> Radius (optional)
    """

    def __init__(self, path):
        self.path = path
        self.type_params = {}
        self.element_to_types = {}
        if path:
            self._load(path)

    @staticmethod
    def _guess_element_from_type(type_label):
        el = ""
        for c in type_label:
            if c.isalpha():
                el += c
            else:
                break
        return el

    def _load(self, path):
        lower = path.lower()
        if lower.endswith(".py"):
            self._load_from_py_dict(path)
        else:
            self._load_from_table(path)

    def _load_from_py_dict(self, path):
        spec = importlib.util.spec_from_file_location("ff_module", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import forcefield module from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, "UFF_DATA"):
            raise RuntimeError(f"{path} does not define UFF_DATA")

        data = getattr(mod, "UFF_DATA")

        for tlabel, vals in data.items():
            if len(vals) < 6:
                continue
            r1 = float(vals[0])          # bond radius (Ri)
            theta0 = float(vals[1])      # equilibrium valence angle (deg)
            x1 = float(vals[2])          # vdW r_star
            D1 = float(vals[3])          # vdW epsilon
            zeta = float(vals[4])        # not used explicitly
            Z1 = float(vals[5])          # effective charge Z_i
            Vi = float(vals[6]) if len(vals) > 6 else 0.0
            Uj = float(vals[7]) if len(vals) > 7 else 0.0
            chi = float(vals[8]) if len(vals) > 8 else 0.0  # electronegativity
            hard = float(vals[9]) if len(vals) > 9 else 10.0  # hardness
            radius = float(vals[10]) if len(vals) > 10 else 1.0  # radius

            el = self._guess_element_from_type(tlabel)
            self.type_params[tlabel] = {
                "bond": r1,
                "angle": theta0,
                "r_star": x1,
                "epsilon": D1,
                "scale": zeta,
                "Z": Z1,
                "Vi": Vi,
                "Uj": Uj,
                "chi": chi,
                "hard": hard,
                "radius": radius,
                "element": el,
            }
            self.element_to_types.setdefault(el, []).append(tlabel)

    def _load_from_table(self, path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                tlabel = parts[0]
                try:
                    bond = float(parts[1])
                    angle = float(parts[2])
                    r_star = float(parts[3])
                    epsilon = float(parts[4])
                    scale = float(parts[5])
                    q_eff = float(parts[6])
                except ValueError:
                    continue

                el = self._guess_element_from_type(tlabel)
                self.type_params[tlabel] = {
                    "bond": bond,
                    "angle": angle,
                    "r_star": r_star,
                    "epsilon": epsilon,
                    "scale": scale,
                    "Z": q_eff,
                    "Vi": 0.0,
                    "Uj": 0.0,
                    "chi": 0.0,
                    "hard": 10.0,
                    "radius": 1.0,
                    "element": el,
                }
                self.element_to_types.setdefault(el, []).append(tlabel)

    def pick_default_type_for_element(self, element_symbol):
        types = self.element_to_types.get(element_symbol, [])
        if types:
            return types[0]
        return None

    def get(self, type_label, default=None):
        if self.type_params is None:
            return default
        return self.type_params.get(type_label, default)


# -------------------- DDEC6 & QEq charges --------------------


def load_ddec6_xyz(path):
    atoms = []
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        elem = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            q = float(parts[4])
        except ValueError:
            continue
        atoms.append({"element": elem, "x": x, "y": y, "z": z, "q": q})
    return atoms


def map_charges_to_structure(structure, ddec_atoms, tol=0.2):
    nat = len(structure)
    charges = [0.0] * nat

    if len(ddec_atoms) == nat:
        for i in range(nat):
            charges[i] = ddec_atoms[i]["q"]
        return charges

    struct_coords = [site.coords for site in structure.sites]
    used = set()
    for i, coord in enumerate(struct_coords):
        best_j = None
        best_d = 1e9
        for j, da in enumerate(ddec_atoms):
            if j in used:
                continue
            dx = coord[0] - da["x"]
            dy = coord[1] - da["y"]
            dz = coord[2] - da["z"]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= tol:
            charges[i] = ddec_atoms[best_j]["q"]
            used.add(best_j)

    return charges


def compute_qeq_charges(structure, uff: UFFParameters, total_charge=0.0,
                        distance_scale=1.0, min_distance=0.5):
    """
    QEq (charge equilibration) using UFF electronegativities and hardness:
      E = sum_i ( chi_i q_i + 0.5 * eta_i q_i^2 )
          + 0.5 * sum_{i!=j} J_ij q_i q_j
      with neutrality: sum_i q_i = total_charge
    """
    nat = len(structure)
    if nat == 0:
        return []

    species = [site.specie.symbol for site in structure.sites]

    chi = np.zeros(nat, dtype=float)
    eta = np.zeros(nat, dtype=float)

    for i, el in enumerate(species):
        tlabel = uff.pick_default_type_for_element(el) if uff is not None else None
        p = uff.get(tlabel) if (uff is not None and tlabel is not None) else None
        if p is not None:
            chi[i] = float(p.get("chi", 5.0))
            eta[i] = float(p.get("hard", p.get("Hard", 10.0)))
        else:
            chi[i] = 5.0
            eta[i] = 10.0
        if eta[i] <= 1.0e-6:
            eta[i] = 10.0  # avoid singular

    coords = np.array([site.coords for site in structure.sites], dtype=float)

    # Build J_ij
    J = np.zeros((nat, nat), dtype=float)
    for i in range(nat):
        for j in range(i + 1, nat):
            rij = np.linalg.norm(coords[i] - coords[j])
            if rij < min_distance:
                rij = min_distance
            val = distance_scale / rij
            J[i, j] = val
            J[j, i] = val

    # Build A and b
    A = np.zeros((nat + 1, nat + 1), dtype=float)
    A[:nat, :nat] = np.diag(eta) + J
    A[:nat, nat] = 1.0
    A[nat, :nat] = 1.0

    b = np.zeros(nat + 1, dtype=float)
    b[:nat] = -chi
    b[nat] = total_charge

    try:
        x = np.linalg.solve(A, b)
        q = x[:nat]
        return q.tolist()
    except np.linalg.LinAlgError:
        # Fallback: diagonal-only QEq:
        inv_eta = 1.0 / eta
        s1 = np.sum(inv_eta)
        s2 = np.sum(chi * inv_eta)
        lam = (s2 - total_charge) / s1
        q = -(chi + lam) * inv_eta
        return q.tolist()


# -------------------- Topology building --------------------


METAL_ELEMENTS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Th", "U", "Np", "Pu", "Am", "Cm",
}


def build_topology(structure, pair_cutoffs=None, covalent_scale=1.2):
    natoms = len(structure)
    species = [site.specie.symbol for site in structure.sites]
    unique_elems = sorted(set(species))

    cov_r = {}
    for el in unique_elems:
        try:
            r = Element(el).covalent_radius
        except Exception:
            r = None
        if r is None:
            r = 0.7
        cov_r[el] = r

    if pair_cutoffs:
        max_cut = max(pair_cutoffs.values())
    else:
        max_cut = 0.0
        for e1 in unique_elems:
            for e2 in unique_elems:
                cut = covalent_scale * (cov_r[e1] + cov_r[e2])
                if cut > max_cut:
                    max_cut = cut

    all_neighbors = structure.get_all_neighbors(r=max_cut, include_index=True)

    bonds = []
    adjacency = {i: set() for i in range(1, natoms + 1)}

    for i, neighs in enumerate(all_neighbors):
        el_i = species[i]
        for nn in neighs:
            j = nn.index
            if j <= i:
                continue
            el_j = species[j]
            r_ij = nn.nn_distance

            if pair_cutoffs:
                key = tuple(sorted((el_i, el_j)))
                cutoff_ij = pair_cutoffs.get(key)
                if cutoff_ij is None:
                    cutoff_ij = covalent_scale * (cov_r[el_i] + cov_r[el_j])
            else:
                cutoff_ij = covalent_scale * (cov_r[el_i] + cov_r[el_j])

            if r_ij <= cutoff_ij:
                i1 = i + 1
                j1 = j + 1
                bonds.append((i1, j1, r_ij))
                adjacency[i1].add(j1)
                adjacency[j1].add(i1)

    angles = []
    for j in range(1, natoms + 1):
        neighs = sorted(adjacency[j])
        if len(neighs) < 2:
            continue
        for a in range(len(neighs)):
            for b in range(a + 1, len(neighs)):
                i = neighs[a]
                k = neighs[b]
                angles.append((i, j, k))

    dihedrals = set()
    for (a, b, _) in bonds:
        for j, k in [(a, b), (b, a)]:
            for i in adjacency[j]:
                if i == k:
                    continue
                for l in adjacency[k]:
                    if l == j:
                        continue
                    dihedrals.add((i, j, k, l))
    dihedrals = sorted(dihedrals)

    impropers = []
    for j in range(1, natoms + 1):
        neighs = sorted(adjacency[j])
        if len(neighs) < 3:
            continue
        for combo in itertools.combinations(neighs, 3):
            i, k, l = combo
            impropers.append((j, i, k, l))

    return bonds, angles, dihedrals, impropers


def build_adjacency_from_bonds(natoms, bonds):
    adjacency = {i: set() for i in range(1, natoms + 1)}
    for (i, j, _) in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


# -------------------- Geometry helpers with PBC --------------------


def pbc_vector(structure, i, j):
    """
    Minimum-image vector from atom i -> j (PBC aware), in Cartesian coordinates.
    i, j are 1-based indices.
    """
    lattice = structure.lattice
    fi = structure.sites[i - 1].frac_coords
    fj = structure.sites[j - 1].frac_coords
    df = [fj[0] - fi[0], fj[1] - fi[1], fj[2] - fi[2]]
    for k in range(3):
        df[k] -= round(df[k])
    cart = lattice.get_cartesian_coords(df)
    return cart


def compute_angle_deg(structure, i, j, k):
    """
    Angle i-j-k in degrees, with PBC minimum image from central atom j.
    """
    v1 = pbc_vector(structure, j, i)  # j -> i
    v2 = pbc_vector(structure, j, k)  # j -> k

    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    cosang = dot / (norm1 * norm2)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def compute_dihedral_deg(structure, i, j, k, l):
    """
    Dihedral angle i-j-k-l in degrees, PBC-aware.
    """
    b0 = pbc_vector(structure, i, j)
    b1 = pbc_vector(structure, j, k)
    b2 = pbc_vector(structure, k, l)

    def norm(v):
        return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    def dot(v, w):
        return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]

    def cross(v, w):
        return [
            v[1] * w[2] - v[2] * w[1],
            v[2] * w[0] - v[0] * w[2],
            v[0] * w[1] - v[1] * w[0],
        ]

    b1_len = norm(b1)
    if b1_len < 1e-8:
        return 0.0
    b1n = [b1[0] / b1_len, b1[1] / b1_len, b1[2] / b1_len]

    b0_dot = dot(b0, b1n)
    b2_dot = dot(b2, b1n)
    v = [b0[0] - b0_dot * b1n[0],
         b0[1] - b0_dot * b1n[1],
         b0[2] - b0_dot * b1n[2]]
    w = [b2[0] - b2_dot * b1n[0],
         b2[1] - b2_dot * b1n[1],
         b2[2] - b2_dot * b1n[2]]

    x = dot(v, w)
    y = dot(cross(b1n, v), w)

    angle_rad = math.atan2(y, x)
    return math.degrees(angle_rad)


# -------------------- Atom types & fragments --------------------


def classify_guest_component(counts: Counter) -> str:
    """
    Heuristic classification of non-framework (guest) components:
      - Typical inorganic/organic anions -> COUNTERION
      - Everything else -> GUEST
    """
    # Simple halides
    simple_halides = [{"Cl": 1}, {"Br": 1}, {"I": 1}, {"F": 1}]
    for d in simple_halides:
        if counts == Counter(d):
            return "COUNTERION"

    # Nitrate-like: NO3(-), HNO3
    if counts == Counter({"N": 1, "O": 3}) or counts == Counter({"N": 1, "O": 3, "H": 1}):
        return "COUNTERION"

    # Sulfate / bisulfate: SO4(2-), HSO4(-)
    if counts == Counter({"S": 1, "O": 4}) or counts == Counter({"S": 1, "O": 4, "H": 1}):
        return "COUNTERION"

    # Acetate-like: C2H3O2(-) or protonated variants
    if counts.get("C") == 2 and counts.get("O") == 2 and counts.get("H", 0) in (3, 4):
        return "COUNTERION"

    # Otherwise treat as generic guest
    return "GUEST"


def assign_atom_types_and_fragments(structure, bonds):
    natoms = len(structure)
    species = [site.specie.symbol for site in structure.sites]
    adjacency = build_adjacency_from_bonds(natoms, bonds)

    atom_types = [None] * natoms
    fragments = [""] * natoms

    # PASS 1: non-metals basic typing
    for idx in range(1, natoms + 1):
        elem = species[idx - 1]
        neighs = adjacency[idx]
        neigh_elems = [species[j - 1] for j in neighs]
        heavy_neighs = [j for j in neighs if species[j - 1] != "H"]
        heavy_elems = [species[j - 1] for j in heavy_neighs]

        nH = neigh_elems.count("H")
        nC = neigh_elems.count("C")
        nO = neigh_elems.count("O")
        nN = neigh_elems.count("N")
        nP = neigh_elems.count("P")
        nS = neigh_elems.count("S")
        n_metals = sum(1 for e in neigh_elems if e in METAL_ELEMENTS)

        deg = len(neighs)
        deg_heavy = len(heavy_neighs)

        if elem in METAL_ELEMENTS:
            continue

        if elem == "P":
            if nO >= 3 and nC >= 1:
                atom_types[idx - 1] = "P_phosphonate"
            elif nO >= 4 and nC == 0:
                atom_types[idx - 1] = "P_phosphate"
            else:
                atom_types[idx - 1] = "P_generic"
            continue

        if elem == "S":
            if nO >= 3:
                atom_types[idx - 1] = "S_sp3_sulfonate"
            else:
                atom_types[idx - 1] = "S_generic"
            continue

        if elem == "C":
            if deg == 4 and nO == 0:
                atom_types[idx - 1] = "C_sp3_alkyl"
            elif deg == 3:
                if nO >= 1:
                    if nO == 1:
                        atom_types[idx - 1] = "C_sp2_carbonyl"
                    elif nO >= 2:
                        carbox_O_indices = [j for j in neighs if species[j - 1] == "O"]
                        any_O_has_H = False
                        for oj in carbox_O_indices:
                            oj_neighs = adjacency[oj]
                            if any(species[k - 1] == "H" for k in oj_neighs):
                                any_O_has_H = True
                                break
                        if any_O_has_H:
                            atom_types[idx - 1] = "C_sp2_carboxylic_acid"
                        else:
                            atom_types[idx - 1] = "C_sp2_carboxylate"
                else:
                    atom_types[idx - 1] = "C_sp2_aromatic"
            elif deg == 2 and nO >= 1:
                atom_types[idx - 1] = "C_sp2_carbonyl"
            else:
                atom_types[idx - 1] = "C_generic"
            continue

        if elem == "O":
            if nH >= 2 and n_metals == 0 and nC == 0 and nP == 0 and nS == 0:
                atom_types[idx - 1] = "Ow"
            elif nH >= 2 and n_metals > 0:
                atom_types[idx - 1] = "Owc"
            else:
                if nH == 1 and n_metals > 0 and nC == 0 and nP == 0 and nS == 0:
                    atom_types[idx - 1] = "O_hydroxide"
                elif nC == 1:
                    C_idx = [j for j in neighs if species[j - 1] == "C"][0]
                    C_neighs = adjacency[C_idx]
                    C_O_neighs = [k for k in C_neighs if species[k - 1] == "O"]
                    if len(C_O_neighs) >= 2:
                        if nH > 0:
                            atom_types[idx - 1] = "O_sp3_carboxylate"
                        else:
                            atom_types[idx - 1] = "O_sp2_carboxylate"
                    else:
                        atom_types[idx - 1] = "O_sp2_carbonyl"
                elif nP >= 1:
                    if nH > 0:
                        atom_types[idx - 1] = "O_sp3_phosphonate"
                    else:
                        if deg_heavy >= 2:
                            atom_types[idx - 1] = "O_sp2_phosphonate"
                        else:
                            atom_types[idx - 1] = "O_sp3_phosphonate"
                elif nS >= 1:
                    atom_types[idx - 1] = "O_sp3_sulfonate"
                else:
                    atom_types[idx - 1] = "O_generic"
            continue

        if elem == "N":
            atom_types[idx - 1] = "N_generic"
            continue

        if elem == "H":
            atom_types[idx - 1] = "H_generic"
            continue

        if elem in {"F", "Cl", "Br", "I"}:
            atom_types[idx - 1] = f"{elem}_halide"
            continue

        atom_types[idx - 1] = f"{elem}_generic"

    # PASS 2: refine N types
    for idx in range(1, natoms + 1):
        elem = species[idx - 1]
        if elem != "N":
            continue
        neighs = adjacency[idx]
        neigh_elems = [species[j - 1] for j in neighs]
        heavy_neighs = [j for j in neighs if species[j - 1] != "H"]

        nH = neigh_elems.count("H")
        if len(neighs) == 4 and nH >= 3:
            atom_types[idx - 1] = "N_ammonium"
            continue

        aromatic_neigh = False
        for j in heavy_neighs:
            if species[j - 1] == "C" and atom_types[j - 1] and "sp2_aromatic" in atom_types[j - 1]:
                aromatic_neigh = True
                break
        if aromatic_neigh:
            atom_types[idx - 1] = "N_sp2_aromatic_amine"
        else:
            atom_types[idx - 1] = "N_sp3_alkyl_amine"

    # PASS 3: refine H types based on heavy neighbour
    for idx in range(1, natoms + 1):
        elem = species[idx - 1]
        if elem != "H":
            continue
        neighs = adjacency[idx]
        if not neighs:
            atom_types[idx - 1] = "H_generic"
            continue
        j = list(neighs)[0]
        heavy_elem = species[j - 1]
        heavy_type = atom_types[j - 1]

        if heavy_elem == "O":
            if heavy_type == "Ow":
                atom_types[idx - 1] = "Hw"
            elif heavy_type == "Owc":
                atom_types[idx - 1] = "Hwc"
            elif "carboxylate" in heavy_type or "carboxylic" in heavy_type:
                atom_types[idx - 1] = "H_carboxylic_acid"
            elif "sulfonate" in heavy_type:
                atom_types[idx - 1] = "H_sulfonic_acid"
            elif "phosphonate" in heavy_type:
                atom_types[idx - 1] = "H_phosphonic_acid"
            elif heavy_type == "O_hydroxide":
                atom_types[idx - 1] = "H_hydroxide"
            else:
                atom_types[idx - 1] = "H_generic"
        elif heavy_elem == "N":
            if atom_types[j - 1] == "N_ammonium":
                atom_types[idx - 1] = "H_ammonium"
            else:
                atom_types[idx - 1] = "H_amine"
        elif heavy_elem == "C":
            if heavy_type and "sp3_alkyl" in heavy_type:
                atom_types[idx - 1] = "H_alkyl"
            elif heavy_type and "sp2_aromatic" in heavy_type:
                atom_types[idx - 1] = "H_aromatic"
            elif heavy_type and "sp2_carbonyl" in heavy_type:
                atom_types[idx - 1] = "H_carbonyl"
            else:
                atom_types[idx - 1] = "H_generic"
        else:
            atom_types[idx - 1] = "H_generic"

    # PASS 4: metals environment label
    for idx in range(1, natoms + 1):
        elem = species[idx - 1]
        if elem not in METAL_ELEMENTS:
            continue

        neighs = adjacency[idx]
        ligand_heavy = [j for j in neighs if species[j - 1] != "H" and species[j - 1] not in METAL_ELEMENTS]
        CN = len(ligand_heavy)

        if CN == 8:
            base = "M_octacoordinated"
        elif CN == 7:
            base = "M_heptacoordinated"
        elif CN == 6:
            base = "M_hexacoordinated"
        elif CN == 5:
            base = "M_pentacoordinated"
        elif CN == 4:
            base = "M_tetracoordinated"
        elif CN == 3:
            base = "M_tricoordinated"
        elif CN == 2:
            base = "M_dicoordinated"
        elif CN == 1:
            base = "M_monocoordinated"
        else:
            base = "M_uncoordinated"

        ligand_classes = []
        for j in ligand_heavy:
            l_type = atom_types[j - 1] or species[j - 1]
            mu = sum(1 for k in adjacency[j] if species[k - 1] in METAL_ELEMENTS)
            ligand_classes.append((l_type, mu))

        env_counts = Counter(ligand_classes)

        if env_counts:
            parts = []
            for (lt, mu), c in sorted(env_counts.items(), key=lambda x: (x[0][0], x[0][1])):
                parts.append(f"{lt}_mu{mu}x{c}")
            env_str = "__".join(parts)
            t = f"{elem}_{base}_{env_str}"
        else:
            t = f"{elem}_{base}"

        atom_types[idx - 1] = t

    # PASS 5: fragment classification (SBU / LINKER / GUEST / COUNTERION)

    # 5a. Determine framework set by BFS starting from metals
    metal_nodes = {i for i in range(1, natoms + 1) if species[i - 1] in METAL_ELEMENTS}
    framework_mask = [False] * natoms

    from collections import deque
    queue = deque()
    for m in metal_nodes:
        queue.append(m)
        framework_mask[m - 1] = True

    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if not framework_mask[v - 1]:
                framework_mask[v - 1] = True
                queue.append(v)

    # 5b. Assign SBU vs LINKER within framework; non-framework → temporary GUEST
    for idx in range(1, natoms + 1):
        if framework_mask[idx - 1]:
            elem = species[idx - 1]
            neighs = adjacency[idx]
            if elem in METAL_ELEMENTS or any(species[j - 1] in METAL_ELEMENTS for j in neighs):
                fragments[idx - 1] = "SBU"
            else:
                fragments[idx - 1] = "LINKER"
        else:
            fragments[idx - 1] = "GUEST"

    # 5c. For non-framework atoms: identify guest components and mark COUNTERIONs
    guest_indices = [i for i in range(1, natoms + 1) if not framework_mask[i - 1]]
    visited_guest = set()

    for start in guest_indices:
        if start in visited_guest:
            continue
        comp = []
        stack = [start]
        visited_guest.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adjacency[u]:
                if (not framework_mask[v - 1]) and (v not in visited_guest):
                    visited_guest.add(v)
                    stack.append(v)

        counts = Counter(species[i - 1] for i in comp)
        gclass = classify_guest_component(counts)
        if gclass == "COUNTERION":
            for i in comp:
                fragments[i - 1] = "COUNTERION"
        else:
            for i in comp:
                fragments[i - 1] = "GUEST"

    return atom_types, fragments


def analyze_linker_environments(structure, bonds, atom_types):
    """
    Old environment analysis helper (kept for potential future use).
    Not critical for the new fragment flags.
    """
    natoms = len(structure)
    species = [site.specie.symbol for site in structure.sites]
    adjacency = build_adjacency_from_bonds(natoms, bonds)

    metal_nodes = {i for i in range(1, natoms + 1) if species[i - 1] in METAL_ELEMENTS}

    adjacency_nm = {i: set() for i in range(1, natoms + 1) if i not in metal_nodes}
    for i in adjacency_nm.keys():
        for j in adjacency[i]:
            if j in adjacency_nm:
                adjacency_nm[i].add(j)

    visited = set()
    components = []
    for i in adjacency_nm.keys():
        if i in visited:
            continue
        comp = []
        stack = [i]
        visited.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adjacency_nm[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(comp)

    comp_infos = []

    for comp in components:
        has_metal_neighbor = False
        metal_coord_counts = {}
        for i in comp:
            metal_neigh = [j for j in adjacency[i] if j in metal_nodes]
            metal_coord_counts[i] = len(metal_neigh)
            if metal_neigh:
                has_metal_neighbor = True

        comp_species = [species[i - 1] for i in comp]
        comp_counts = Counter(comp_species)

        if has_metal_neighbor:
            classification = "LINKER"
        else:
            if all(e in {"H", "C", "N", "O"} for e in comp_species) and len(comp) <= 8:
                classification = "SOLVENT"
            elif any(e in {"Cl", "Br", "I", "F", "S", "P"} for e in comp_species) and len(comp) <= 10:
                classification = "COUNTER_ION"
            else:
                classification = "MOLECULE"

        comp_info = {
            "atoms": comp,
            "metal_coord_counts": metal_coord_counts,
            "classification": classification,
            "species_counts": comp_counts,
        }
        comp_infos.append(comp_info)

    groups = {}
    for ci in comp_infos:
        classification = ci["classification"]
        counts = ci["species_counts"]
        comp_key = classification + ":" + "|".join(f"{el}{counts[el]}" for el in sorted(counts.keys()))
        if comp_key not in groups:
            groups[comp_key] = {
                "env_key": comp_key,
                "classification": classification,
                "components": [],
                "label": None,
            }
        groups[comp_key]["components"].append(ci)

    return comp_infos, groups


# -------------------- Lattice conversion --------------------


def lattice_to_lammps_box(lattice):
    m = lattice.matrix
    a = m[0]
    b = m[1]
    c = m[2]

    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    xlo = 0.0
    ylo = 0.0
    zlo = 0.0

    a_len = math.sqrt(ax * ax + ay * ay + az * az)
    if a_len == 0:
        raise ValueError("Lattice vector a has zero length.")

    ax_hat = ax / a_len
    ay_hat = ay / a_len
    az_hat = az / a_len

    xhi = a_len

    xy = bx * ax_hat + by * ay_hat + bz * az_hat
    xz = cx * ax_hat + cy * ay_hat + cz * az_hat

    b_sq = bx * bx + by * by + bz * bz
    yy_sq = b_sq - xy * xy
    if yy_sq < 0:
        yy_sq = 0.0
    yy = math.sqrt(yy_sq)

    if yy == 0:
        yz = 0.0
    else:
        b_dot_c = bx * cx + by * cy + bz * cz
        yz = (b_dot_c - xy * xz) / yy

    c_sq = cx * cx + cy * cy + cz * cz
    zz_sq = c_sq - xz * xz - yz * yz
    if zz_sq < 0:
        zz_sq = 0.0
    zz = math.sqrt(zz_sq)

    xhi_val = xhi
    yhi_val = yy
    zhi_val = zz

    return xlo, xhi_val, ylo, yhi_val, zlo, zhi_val, xy, xz, yz


# -------------------- LAMMPS data writer --------------------


def write_lammps_data(
    structure,
    bonds,
    angles,
    dihedrals,
    impropers,
    uff: UFFParameters,
    charges,
    chem_atom_types=None,
    fragments=None,
    ff_mode="UFF_all_cryst",
    use_spcfw=False,
    include_dihedrals=True,
    include_impropers=True,
    restrain_bond_factor=0.0,
    restrain_angle_factor=0.0,
    restrain_policy="soften_mismatch",
    title="LAMMPS data file",
):
    """
    ff_mode:
      - 'UFF_all_cryst'  : UFF with crystallographic equilibrium geometry (CIF).
                           r0/θ0 are taken from CIF (averaged per type). Force constants can be
                           modified using the restraint sliders (restrain_*_factor) and the selected
                           restraint policy (restrain_policy).
      - 'UFF_all_uff'    : Pure UFF (force constants + equilibrium r0/θ0 from UFF).

    restrain_policy (used only when ff_mode == 'UFF_all_cryst' and sliders > 0):
      - 'soften_mismatch': k is reduced with mismatch between CIF and UFF minima (never stiffens)
      - 'recompute_cif'  : recompute k using CIF equilibrium geometry with the same UFF formulas
                           (can soften or stiffen, e.g., shorter CIF bonds → larger k)
    """
    ff_mode_effective = ff_mode

    natoms = len(structure)
    species = [site.specie.symbol for site in structure.sites]

    if chem_atom_types is None or len(chem_atom_types) != natoms:
        chem_atom_types = [species[i] for i in range(natoms)]
    if fragments is None or len(fragments) != natoms:
        fragments = ["GEN" for _ in range(natoms)]

    # SPC/Fw parameters
    spcfw_eps_O = 0.1554253
    spcfw_sig_O = 3.165492
    spcfw_eps_H = 0.0
    spcfw_sig_H = 1.0
    spcfw_qO = -0.82
    spcfw_qH = 0.41
    spcfw_k_bond = 1059.162
    spcfw_r0 = 1.012
    spcfw_k_angle = 75.90
    spcfw_theta0 = 113.24

    # LAMMPS atom type labels
    atom_lmp_labels = list(chem_atom_types)

    type_element = {}
    for i in range(natoms):
        lbl = atom_lmp_labels[i]
        el = species[i]
        if lbl not in type_element:
            type_element[lbl] = el

    unique_types = []
    type_id = {}
    for lbl in atom_lmp_labels:
        if lbl not in type_id:
            type_id[lbl] = len(type_id) + 1
            unique_types.append(lbl)

    # Remove metal dihedrals/impropers first
    filtered_dihedrals = []
    for (i, j, k, l) in dihedrals:
        if any(species[idx - 1] in METAL_ELEMENTS for idx in (i, j, k, l)):
            continue
        filtered_dihedrals.append((i, j, k, l))
    dihedrals = filtered_dihedrals

    filtered_impropers = []
    for (j, i, k, l) in impropers:
        if any(species[idx - 1] in METAL_ELEMENTS for idx in (i, j, k, l)):
            continue
        filtered_impropers.append((j, i, k, l))
    impropers = filtered_impropers

    if not include_dihedrals:
        dihedrals = []
    if not include_impropers:
        impropers = []

    nbonds = len(bonds)
    nangles = len(angles)

    # ---- Bond types ----
    bond_type_map = {}
    bond_types = []
    for (i, j, dist) in bonds:
        ti = atom_lmp_labels[i - 1]
        tj = atom_lmp_labels[j - 1]
        key = tuple(sorted((ti, tj)))
        if key not in bond_type_map:
            bond_type_map[key] = len(bond_type_map) + 1
            bond_types.append(key)
    nbondtypes = len(bond_types)

    bond_type_samples = {key: [] for key in bond_types}
    for (i, j, dist) in bonds:
        ti = atom_lmp_labels[i - 1]
        tj = atom_lmp_labels[j - 1]
        key = tuple(sorted((ti, tj)))
        if key in bond_type_samples:
            bond_type_samples[key].append(dist)
    bond_type_r0_cif = {}
    for key, ds in bond_type_samples.items():
        if ds:
            bond_type_r0_cif[key] = sum(ds) / len(ds)
        else:
            bond_type_r0_cif[key] = 1.5

    # ---- Angle types ----
    angle_type_map = {}
    angle_type_samples = {}
    angle_instances = []

    type_el = type_element

    for (i, j, k) in angles:
        ti = atom_lmp_labels[i - 1]
        tj = atom_lmp_labels[j - 1]
        tk = atom_lmp_labels[k - 1]
        theta = compute_angle_deg(structure, i, j, k)

        el_j = type_el[tj]
        base_key = (ti, tj, tk)

        if el_j in METAL_ELEMENTS:
            theta_bin = int(round(theta / 5.0) * 5)
            key = base_key + (f"{theta_bin}deg",)
        else:
            key = base_key

        angle_type_samples.setdefault(key, []).append(theta)
        angle_instances.append((i, j, k, key))

    angle_type_theta0_cif = {}
    for key in sorted(angle_type_samples.keys(), key=lambda x: (x[1], x[0], x[2])):
        type_idx = len(angle_type_map) + 1
        angle_type_map[key] = type_idx
        samples = angle_type_samples[key]
        theta0 = sum(samples) / len(samples)
        angle_type_theta0_cif[key] = theta0
    nangletypes = len(angle_type_map)

    # ---- Adjacency (needed for torsion degeneracy etc.) ----
    adjacency = build_adjacency_from_bonds(natoms, bonds)

    # ---- UFF element→type mapping (per element, for LJ & QEq style things) ----
    unique_elems = sorted(set(species))
    element_to_uff_type = {}
    if uff is not None:
        for el in unique_elems:
            tlabel = uff.pick_default_type_for_element(el)
            element_to_uff_type[el] = tlabel
    else:
        element_to_uff_type = {el: None for el in unique_elems}

    # ---- UFF helper functions ----
    LAMBDA_BO = 0.1332
    KR_PREF = 664.12

    def _uff_get_for_element(el):
        if uff is None:
            return None
        tlabel = element_to_uff_type.get(el)
        if not tlabel:
            return None
        return uff.get(tlabel)

    # Helper to strip fragment suffix (_LINKER/_SBU/_GUEST/_COUNTERION/_GEN)
    def strip_fragment_suffix(lbl):
        for suf in ("_LINKER", "_SBU", "_GUEST", "_COUNTERION", "_GEN"):
            if lbl.endswith(suf):
                return lbl[: -len(suf)]
        return lbl

    # Map our chem labels -> UFF type labels (C_3, C_R, O_3, O_2, P_3+q, S_3+6, H_, Zn3+2, ...)
    def guess_uff_type_for_label(lbl, element):
        core = strip_fragment_suffix(lbl)

        if element == "C":
            if "aromatic" in core:
                return "C_R"
            if "sp2" in core:
                return "C_2"
            return "C_3"

        if element == "N":
            if "aromatic" in core:
                return "N_R"
            if "sp2" in core:
                return "N_2"
            return "N_3"

        if element == "O":
            if "sp2" in core or "carbonyl" in core:
                return "O_2"
            return "O_3"

        if element == "P":
            # all phosphonates/phosphates as P_3+q
            return "P_3+q"

        if element == "S":
            if "sulfonate" in core or "sulfate" in core:
                return "S_3+6"
            return "S_3"

        if element == "H":
            return "H_"

        if element == "Zn":
            return "Zn3+2"

        # fallback: let element-based mapping decide
        return None

    # Build atom-type → UFF-type mapping
    atom_to_uff_type = {}
    if uff is not None and uff.type_params:
        for lbl in unique_types:
            el = type_element[lbl]
            t_guess = guess_uff_type_for_label(lbl, el)
            if t_guess is not None and t_guess in uff.type_params:
                atom_to_uff_type[lbl] = t_guess
            else:
                atom_to_uff_type[lbl] = element_to_uff_type.get(el)
    else:
        atom_to_uff_type = {lbl: None for lbl in unique_types}

    def _uff_get_for_label(lbl):
        """
        Get UFF parameter dict for a given LAMMPS atom label
        (C_sp3_alkyl_LINKER, etc.) via underlying UFF type (C_3, C_R, ...).
        """
        if uff is None:
            return None
        t = atom_to_uff_type.get(lbl)
        if not t:
            return None
        return uff.get(t)

    def _uff_bond_params_for_labels(lbl_i, lbl_j, bond_order=1.0):
        el_i = type_element[lbl_i]
        el_j = type_element[lbl_j]
        p_i = _uff_get_for_label(lbl_i)
        p_j = _uff_get_for_label(lbl_j)
        if p_i is None or p_j is None or uff is None:
            return None, None

        Ri = p_i["bond"]
        Rj = p_j["bond"]
        Zi = p_i["Z"]
        Zj = p_j["Z"]
        chi_i = p_i.get("chi", 0.0)
        chi_j = p_j.get("chi", 0.0)

        denom = chi_i * Ri + chi_j * Rj
        if denom <= 0.0 or chi_i <= 0.0 or chi_j <= 0.0:
            r_en = 0.0
        else:
            term = (math.sqrt(chi_i) - math.sqrt(chi_j)) ** 2
            r_en = (Ri * Rj * term) / denom

        if bond_order <= 0.0:
            r_bo = 0.0
        else:
            r_bo = -LAMBDA_BO * (Ri + Rj) * math.log(bond_order)

        R_ij = (Ri + Rj) + r_bo - r_en
        if R_ij <= 0.0:
            return None, None

        k_r = KR_PREF * Zi * Zj / (R_ij ** 3)

        # UFF: U = 1/2 * k_r * (r - r_ij)^2
        # LAMMPS harmonic: U = K * (r - r0)^2
        # => K_LAMMPS = 0.5 * k_r
        k_r *= 0.5

        return k_r, R_ij

    def _uff_angle_params_for_labels(lbl_i, lbl_j, lbl_k,
                                     theta0_deg_override=None,
                                     bond_order_ij=1.0,
                                     bond_order_jk=1.0,
                                     r0_ij_override=None,
                                     r0_jk_override=None):
        """
        UFF valence angle term (eq. 13 με το σωστό bracket
        3 R_ij R_jk (1 - cos^2 θ0) - R_ik^2 cos θ0) χαρτογραφημένο σε
        LAMMPS angle_style harmonic.

        UFF:    E = 1/2 * K_ijk * (θ - θ0)^2
        LAMMPS: E = K_lmp * (θ - θ0)^2
        =>      K_lmp = 0.5 * K_ijk
        """
        el_i = type_element[lbl_i]
        el_j = type_element[lbl_j]
        el_k = type_element[lbl_k]

        p_i = _uff_get_for_label(lbl_i)
        p_j = _uff_get_for_label(lbl_j)
        p_k = _uff_get_for_label(lbl_k)
        if p_i is None or p_j is None or p_k is None or uff is None:
            return None, None

        Zi = p_i["Z"]
        Zk = p_k["Z"]

        # ισορροπίες δεσμών από τον UFF όρο δεσμού
        _, R_ij = _uff_bond_params_for_labels(lbl_i, lbl_j, bond_order_ij)
        _, R_jk = _uff_bond_params_for_labels(lbl_j, lbl_k, bond_order_jk)
        if R_ij is None or R_jk is None:
            return None, None

        # Optional overrides (used for CIF-restrained recomputation of k):
        # If provided, treat the overridden values as the equilibrium bond lengths used in the UFF angle expression.
        if r0_ij_override is not None and r0_ij_override > 1.0e-12:
            R_ij = r0_ij_override
        if r0_jk_override is not None and r0_jk_override > 1.0e-12:
            R_jk = r0_jk_override

        # γωνία ισορροπίας: είτε override, είτε από τα UFF atom params
        if theta0_deg_override is not None:
            theta0_deg = theta0_deg_override
        else:
            theta0_deg = p_j["angle"]

        theta0_rad = math.radians(theta0_deg)
        cos0 = math.cos(theta0_rad)

        # R_ik στην γεωμετρία ισορροπίας (νόμος συνημιτόνου)
        R_ik_sq = R_ij**2 + R_jk**2 - 2.0 * R_ij * R_jk * cos0
        if R_ik_sq <= 0.0:
            return None, None
        R_ik = math.sqrt(R_ik_sq)

        # ΣΩΣΤΟΣ UFF όρος:
        #   3 R_ij R_jk (1 - cos^2 θ0) - R_ik^2 cos θ0
        num = 3.0 * R_ij * R_jk * (1.0 - cos0 * cos0) - cos0 * R_ik_sq
        denom = R_ik**5
        if denom <= 0.0:
            return None, None

        # K_ijk από UFF (σε kcal/mol/rad^2)
        k_ijk = KR_PREF * Zi * Zk * (num / denom)

        # UFF: 1/2 * K_ijk (θ - θ0)^2  → LAMMPS: K_lmp (θ - θ0)^2
        k_lmp = k_ijk

        return k_lmp, theta0_deg

    # ---- Helpers for torsions and impropers ----

    def core_type(lbl):
        """
        Strip last fragment flag (_LINKER/_SBU/_GUEST/_COUNTERION/_GEN)
        to get core chemical type.
        """
        parts = lbl.split("_")
        if len(parts) > 1 and parts[-1] in ("LINKER", "SBU", "GUEST", "COUNTERION", "GEN"):
            return "_".join(parts[:-1])
        return lbl

    def infer_hybrid(lbl, el):
        t = lbl
        if "sp2_aromatic" in t or "aromatic" in t:
            return "sp2"
        if "sp2" in t:
            return "sp2"
        if "sp3" in t:
            return "sp3"
        if el == "C":
            return "sp3"
        if el in {"N", "O", "P", "S"}:
            return "sp3"
        return "sp3"

    def _uff_dihedral_params_for_bond(j_label, k_label, deg_j, deg_k):
        """
        UFF torsion mapping to LAMMPS 'dihedral_style harmonic'
        following the UFF functional form and the way LAMMPS-Interface
        handles torsional degeneracy.

        UFF torsion (simplified) is:
            E_phi(UFF) = 1/2 * V_phi * [1 - cos(n * phi0) * cos(n * phi)]

        LAMMPS 'dihedral_style harmonic' uses:
            E_lmp = K * [1 + d * cos(n * phi)]

        Mapping:
            K = (V_phi_eff / 2)
            d = -cos(n * phi0)
            n = 1, 2, or 3 (fold)

        with torsional degeneracy:
            V_phi_eff = V_phi / ((deg_j - 1) * (deg_k - 1))

        where deg_j, deg_k are the total numbers of neighbors of the
        inner atoms j and k (including H), and we subtract 1 to
        exclude the bond j-k itself.

        This reproduces the small K-values you see in LAMMPS-Interface:
        e.g. H-C(sp3)-C(sp3)-H: 9-fold degeneracy (3×3), and
        P-C(sp3)-N(sp3)-C: 6-fold degeneracy (3×2).
        """
        el_j = type_element[j_label]
        el_k = type_element[k_label]

        # No torsions involving metals
        if el_j in METAL_ELEMENTS or el_k in METAL_ELEMENTS:
            return 0.0, 1.0, 1

        p_j = _uff_get_for_label(j_label)
        p_k = _uff_get_for_label(k_label)
        if p_j is None or p_k is None:
            return 0.0, 1.0, 1

        hyb_j = infer_hybrid(j_label, el_j)
        hyb_k = infer_hybrid(k_label, el_k)

        def is_sp2_like(h):
            return h == "sp2"

        all_sp3 = (hyb_j == "sp3" and hyb_k == "sp3")
        all_sp2 = (is_sp2_like(hyb_j) and is_sp2_like(hyb_k))
        mixed = (is_sp2_like(hyb_j) and hyb_k == "sp3") or (
            hyb_j == "sp3" and is_sp2_like(hyb_k)
        )

        V_phi = 0.0
        n = 1
        phi0 = 0.0

        # --- sp3-sp3 case ---
        if all_sp3:
            # standard UFF: 3-fold, minima at 60° + 120°k
            n = 3
            phi0 = 60.0

            Vi = p_j.get("Vi", 0.0)
            Vj = p_k.get("Vi", 0.0)

            # Heteroatom corrections (UFF)
            # (around X–O or X–S bonds etc.)
            def adjust(el, V_in, n_in, phi_in):
                if el == "O":
                    # two-fold barrier, phi0 = 90°
                    return 2.0, 2, 90.0
                if el in ("S", "Se", "Te", "Po"):
                    # hypervalent S, etc.
                    return 6.8, 2, 90.0
                return V_in, n_in, phi_in

            Vi, n, phi0 = adjust(el_j, Vi, n, phi0)
            Vj, n, phi0 = adjust(el_k, Vj, n, phi0)

            V_phi = math.sqrt(max(Vi, 0.0) * max(Vj, 0.0))

        # --- sp2-sp2 case ---
        elif all_sp2:
            # π-conjugated systems: 2-fold barrier, planar minimum
            n = 2
            phi0 = 180.0

            Ui = p_j.get("Uj", 0.0)
            Uj_ = p_k.get("Uj", 0.0)

            if Ui > 0.0 and Uj_ > 0.0:
                torsiontype = 1.0  # could be modified for strong conjugation
                V_phi = 5.0 * math.sqrt(Ui * Uj_) * (1.0 + 4.18 * math.log(torsiontype))
            else:
                V_phi = 0.0

        # --- mixed sp2/sp3 case ---
        elif mixed:
            # UFF default for mixed: V_phi = 2.0 kcal/mol, 3-fold
            n = 3
            phi0 = 180.0
            V_phi = 2.0

        # --- other cases: no torsion ---
        else:
            V_phi = 0.0

        if V_phi <= 0.0:
            return 0.0, 1.0, 1

        # ---------- DEGENERACY SCALING (UFF / LAMMPS-Interface) ----------
        # Number of distinct dihedrals about bond j-k:
        #  (deg_j - 1) * (deg_k - 1)
        # where deg_j, deg_k include all neighbors (H as well), and
        # we subtract the bond partner itself.
        Nj = max(deg_j - 1, 1)
        Nk = max(deg_k - 1, 1)
        deg_factor = Nj * Nk

        V_phi_eff = V_phi / float(deg_factor)

        # ---------- MAP TO LAMMPS 'dihedral_style harmonic' ----------
        # UFF: E = 1/2 * V_phi_eff * [1 - cos(n*phi0) * cos(n*phi)]
        # LAMMPS: E = K * [1 + d * cos(n*phi)]
        # => K = V_phi_eff / 2,  d = -cos(n * phi0)
        cos_nphi0 = math.cos(math.radians(n * phi0))
        d = -cos_nphi0
        K = 0.5 * V_phi_eff

        return K, d, n

    def _uff_improper_params(center_label, neighbor_labels):
        """
        Approximate UFF out-of-plane (improper) mapping to LAMMPS 'fourier':
           E = K [C0 + C1 cos χ + C2 cos 2χ]
        """
        elc = type_element[center_label]
        p_c = _uff_get_for_label(center_label)
        if p_c is None:
            return 0.0, 1.0, 0.0, 0.0

        if elc not in {"C", "N", "O", "P", "As", "Sb", "Bi"}:
            return 0.0, 1.0, 0.0, 0.0

        Uc = p_c.get("Uj", 0.0)
        if Uc <= 0.0:
            K = 0.0
        else:
            K = Uc

        C0 = 1.0
        C1 = -1.0
        C2 = 0.0

        if elc in {"P", "As", "Sb", "Bi"}:
            if elc == "P":
                phi_deg = 84.4339
            elif elc == "As":
                phi_deg = 86.9735
            elif elc == "Sb":
                phi_deg = 87.7047
            else:
                phi_deg = 90.0
            phi = math.radians(phi_deg)
            C1 = -4.0 * math.cos(phi)
            C2 = 1.0
            C0 = -1.0 * C1 * math.cos(phi) + C2 * math.cos(2.0 * phi)

        if K <= 0.0:
            return 0.0, 1.0, 0.0, 0.0
        return K, C0, C1, C2

    # ---- Dihedral types ----
    dihedral_type_map = {}
    dihedral_type_params = {}
    dihedral_type_repr = {}
    dihedral_instances_typed = []

    if include_dihedrals and dihedrals:
        for (i, j, k, l) in dihedrals:
            ti = atom_lmp_labels[i - 1]
            tj = atom_lmp_labels[j - 1]
            tk = atom_lmp_labels[k - 1]
            tl = atom_lmp_labels[l - 1]

            key_core = (core_type(ti), core_type(tj), core_type(tk), core_type(tl))

            if key_core not in dihedral_type_map:
                dtype = len(dihedral_type_map) + 1
                dihedral_type_map[key_core] = dtype
                dihedral_type_repr[key_core] = (ti, tj, tk, tl)

            deg_j = len(adjacency[j])
            deg_k = len(adjacency[k])
            K, d, n = _uff_dihedral_params_for_bond(tj, tk, deg_j, deg_k)

            if K is None or K <= 0.0:
                K = 0.0
                d_int = 1
                n_int = 1
            else:
                d_int = 1 if d >= 0.0 else -1
                n_int = int(round(n))
                if n_int < 0:
                    n_int = 0

            dihedral_type_params[key_core] = (K, d_int, n_int)
            dihedral_instances_typed.append((i, j, k, l, dihedral_type_map[key_core]))

    dihedral_instances = dihedral_instances_typed
    ndihedrals = len(dihedral_instances)
    ndihedraltypes = len(dihedral_type_map) if include_dihedrals else 0

    # ---- Improper types ----
    improper_type_map = {}
    improper_type_params = {}
    improper_type_repr = {}
    improper_instances_typed = []

    if include_impropers and impropers:
        for (j, i, k, l) in impropers:
            tj = atom_lmp_labels[j - 1]
            ti = atom_lmp_labels[i - 1]
            tk = atom_lmp_labels[k - 1]
            tl = atom_lmp_labels[l - 1]

            cj = core_type(tj)
            ni = core_type(ti)
            nk = core_type(tk)
            nl = core_type(tl)
            neighbors_core_sorted = tuple(sorted((ni, nk, nl)))
            key_core = (cj, neighbors_core_sorted)

            if key_core not in improper_type_map:
                itype = len(improper_type_map) + 1
                improper_type_map[key_core] = itype
                improper_type_repr[key_core] = (tj, ti, tk, tl)

                K, C0, C1, C2 = _uff_improper_params(tj, [ti, tk, tl])
                if K is None or K < 0.0:
                    K = 0.0
                    C0, C1, C2 = 1.0, 0.0, 0.0
                improper_type_params[key_core] = (K, C0, C1, C2)

            itype = improper_type_map[key_core]
            improper_instances_typed.append((j, i, k, l, itype))

    improper_instances = improper_instances_typed
    nimpropers = len(improper_instances)
    nimpropertypes = len(improper_type_map) if include_impropers else 0

    # ---- Box ----
    lattice = structure.lattice
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = lattice_to_lammps_box(lattice)

    # SPC/Fw helpers
    def is_spcfw_O(label, element):
        if element != "O":
            return False
        return ("Ow" in label) or ("Owc" in label)

    def is_spcfw_H(label, element):
        if element != "H":
            return False
        return ("Hw" in label) or ("Hwc" in label)

    def is_spcfw_water_atom(label, element):
        return is_spcfw_O(label, element) or is_spcfw_H(label, element)

    restrain_bond_factor = max(0.0, min(1.0, restrain_bond_factor))
    restrain_angle_factor = max(0.0, min(1.0, restrain_angle_factor))

    # ---- Header ----
    lines = []
    lines.append(f"{title}")
    lines.append("")
    lines.append(f"{natoms:8d} atoms")
    lines.append(f"{nbonds:8d} bonds")
    lines.append(f"{nangles:8d} angles")
    lines.append(f"{ndihedrals:8d} dihedrals")
    lines.append(f"{nimpropers:8d} impropers")
    lines.append("")
    lines.append(f"{len(unique_types):8d} atom types")
    lines.append(f"{nbondtypes:8d} bond types")
    lines.append(f"{nangletypes:8d} angle types")
    lines.append(f"{ndihedraltypes:8d} dihedral types")
    lines.append(f"{nimpropertypes:8d} improper types")
    lines.append("")
    lines.append(f"{xlo: .6f} {xhi: .6f} xlo xhi")
    lines.append(f"{ylo: .6f} {yhi: .6f} ylo yhi")
    lines.append(f"{zlo: .6f} {zhi: .6f} zlo zhi")
    lines.append(f"{xy: .6f} {xz: .6f} {yz: .6f} xy xz yz")
    lines.append("")

    # ---- Masses ----
    lines.append("Masses")
    lines.append("")
    for lbl in unique_types:
        tid = type_id[lbl]
        el_symbol = type_element[lbl]
        try:
            mass = float(Element(el_symbol).atomic_mass)
        except Exception:
            mass = 1.0
        lines.append(f"{tid:5d}  {mass: .6f}  # {lbl}")
    lines.append("")

    # ---- Pair Coeffs ----
    lines.append("# Pair Coeffs (epsilon [kcal/mol], sigma [Å])")
    lines.append("Pair Coeffs")
    lines.append("")
    for lbl in unique_types:
        tid = type_id[lbl]
        el = type_element[lbl]

        if lbl.upper().startswith("H_PROTON"):
            epsilon = 1.0e-6  # kcal/mol
            sigma = 1.238     # Å
            comment = "free proton (charges from DDEC6 / QEq)"
        else:
            uff_label = element_to_uff_type.get(el)
            p = uff.get(uff_label, None) if (uff is not None and uff_label is not None) else None
            if p is not None:
                r_star = p["r_star"]
                epsilon = p["epsilon"]
                sigma = r_star / (2.0 ** (1.0 / 6.0))
            else:
                epsilon = 0.1
                sigma = 3.5

            if use_spcfw and is_spcfw_water_atom(lbl, el):
                if is_spcfw_O(lbl, el):
                    epsilon = spcfw_eps_O
                    sigma = spcfw_sig_O
                    comment = "SPC/Fw O"
                else:
                    epsilon = spcfw_eps_H
                    sigma = spcfw_sig_H
                    comment = "SPC/Fw H"
            else:
                comment = f"(el={el}, UFF={uff_label})"

        lines.append(f"{tid:5d}  {epsilon: .6f}  {sigma: .6f}  # {lbl} {comment}")
    lines.append("")

    # ---- Bond Coeffs ----
    lines.append("# Bond Coeffs (k [kcal/mol/Å^2], r0 [Å])")
    lines.append("Bond Coeffs")
    lines.append("")

    def is_water_label(lbl):
        el = type_element[lbl]
        return is_spcfw_water_atom(lbl, el)

    for idx, key in enumerate(bond_types, start=1):
        ti, tj = key
        el_i = type_element[ti]
        el_j = type_element[tj]

        k_r_uff, R_ij_uff = _uff_bond_params_for_labels(ti, tj, bond_order=1.0)

        # Equilibrium bond length selection:
        # - Pure UFF: use UFF r0
        # - CIF mode: use CIF r0 only if the bond restraint slider is active (>0); otherwise behave as pure UFF
        if (ff_mode_effective == "UFF_all_uff") or (ff_mode_effective == "UFF_all_cryst" and restrain_bond_factor <= 0.0):
            r0 = R_ij_uff if (R_ij_uff is not None) else bond_type_r0_cif.get(key, 1.5)
        else:
            r0 = bond_type_r0_cif.get(key, R_ij_uff if R_ij_uff is not None else 1.5)

        if k_r_uff is not None:
            k = k_r_uff
        else:
            k = 300.0  # fallback

        if (ff_mode_effective == "UFF_all_cryst" and
                k_r_uff is not None and
                R_ij_uff is not None and
                restrain_bond_factor > 0.0):
            r0_cif = bond_type_r0_cif.get(key, R_ij_uff)
            if r0_cif is not None and r0_cif > 1.0e-6 and R_ij_uff > 1.0e-6:
                s = restrain_bond_factor
                policy = (restrain_policy or "soften_mismatch").strip()

                if policy == "recompute_cif":
                    # Recompute stiffness using the same UFF formula, but evaluated at the CIF equilibrium length.
                    # This can soften OR stiffen (e.g., shorter CIF bonds → larger k).
                    p_i = _uff_get_for_label(ti)
                    p_j = _uff_get_for_label(tj)
                    if p_i is not None and p_j is not None:
                        Zi = p_i.get("Z", None)
                        Zj = p_j.get("Z", None)
                    else:
                        Zi = Zj = None

                    if Zi is not None and Zj is not None:
                        k_cif = KR_PREF * Zi * Zj / (r0_cif ** 3)
                        k_cif *= 0.5  # match mapping used in _uff_bond_params_for_labels
                        k_target = k_cif
                    else:
                        k_target = k_r_uff
                else:
                    # Safe policy: attenuate k by mismatch magnitude (never stiffens)
                    delta_rel = abs(r0_cif - R_ij_uff) / R_ij_uff
                    k_target = k_r_uff / (1.0 + delta_rel)

                k = (1.0 - s) * k_r_uff + s * k_target

        comment = f"{ti} -- {tj}"

        if use_spcfw:
            i_is_w = is_water_label(ti)
            j_is_w = is_water_label(tj)
            if i_is_w and j_is_w:
                if ((el_i == "O" and el_j == "H") or
                    (el_i == "H" and el_j == "O")):
                    k = spcfw_k_bond
                    r0 = spcfw_r0
                    comment = f"SPC/Fw {ti} -- {tj}"

        lines.append(f"{idx:5d}  {k: .3f}  {r0: .3f}  # {comment}")
    lines.append("")

    # ---- Angle Coeffs ----
    lines.append("# Angle Coeffs (k [kcal/mol/rad^2], theta0 [deg])")
    lines.append("Angle Coeffs")
    lines.append("")

    for key, type_idx in sorted(angle_type_map.items(), key=lambda x: x[1]):
        if len(key) == 3:
            ti, tj, tk = key
            extra = ""
        else:
            ti, tj, tk, extra = key
            extra = f" {extra}"

        el_j = type_element[tj]
        p_j = _uff_get_for_label(tj)

        theta_uff = None
        if p_j is not None:
            theta_uff = p_j["angle"]

        theta_cif = angle_type_theta0_cif[key]

        # Equilibrium angle selection:
        # - Pure UFF: use UFF theta0
        # - CIF mode: use CIF theta0 only if the angle restraint slider is active (>0); otherwise behave as pure UFF
        if ((ff_mode_effective == "UFF_all_uff") or
                (ff_mode_effective == "UFF_all_cryst" and restrain_angle_factor <= 0.0)) and theta_uff is not None:
            theta0_deg = theta_uff
        else:
            theta0_deg = theta_cif

        if theta_uff is not None:
            k_theta_uff, _ = _uff_angle_params_for_labels(
                ti, tj, tk,
                theta0_deg_override=theta_uff,
                bond_order_ij=1.0,
                bond_order_jk=1.0,
            )
        else:
            k_theta_uff = None

        if k_theta_uff is not None:
            k = k_theta_uff
            theta0 = theta0_deg
        else:
            k = 60.0
            theta0 = theta0_deg

        if (ff_mode_effective == "UFF_all_cryst" and
                k_theta_uff is not None and
                theta_uff is not None and
                abs(theta_uff) > 1.0e-6 and
                restrain_angle_factor > 0.0):
            s = restrain_angle_factor
            policy = (restrain_policy or "soften_mismatch").strip()

            if policy == "recompute_cif":
                # Recompute k using CIF equilibrium geometry with the same UFF angular formula.
                # This may soften or stiffen depending on how CIF geometry differs from UFF.
                key_ij = tuple(sorted((ti, tj)))
                key_jk = tuple(sorted((tj, tk)))
                r0_ij_cif = bond_type_r0_cif.get(key_ij, None)
                r0_jk_cif = bond_type_r0_cif.get(key_jk, None)

                k_theta_cif, _ = _uff_angle_params_for_labels(
                    ti, tj, tk,
                    theta0_deg_override=theta_cif,
                    bond_order_ij=1.0,
                    bond_order_jk=1.0,
                    r0_ij_override=r0_ij_cif,
                    r0_jk_override=r0_jk_cif,
                )
                k_target = k_theta_cif if k_theta_cif is not None else k_theta_uff
            else:
                # Safe policy: attenuate k by mismatch magnitude (never stiffens)
                delta_rel = abs(theta_cif - theta_uff) / abs(theta_uff)
                k_target = k_theta_uff / (1.0 + delta_rel)

            k = (1.0 - s) * k_theta_uff + s * k_target
            theta0 = theta_cif

        comment = f"{ti} -- {tj} -- {tk}{extra}"

        if use_spcfw:
            el_i = type_element[ti]
            el_j = type_element[tj]
            el_k = type_element[tk]
            if (is_spcfw_O(tj, el_j) and
                is_spcfw_H(ti, el_i) and
                is_spcfw_H(tk, el_k)):
                k = spcfw_k_angle
                theta0 = spcfw_theta0
                comment = f"SPC/Fw {ti} -- {tj} -- {tk}{extra}"

        lines.append(f"{type_idx:5d}  {k: .3f}  {theta0: .3f}  # {comment}")
    lines.append("")

    # ---- Dihedral Coeffs ----
    if include_dihedrals and ndihedraltypes > 0:
        lines.append("# Dihedral Coeffs (K [kcal/mol], d, n)  ; dihedral_style harmonic")
        lines.append("Dihedral Coeffs")
        lines.append("")
        for key_core, dtype in sorted(dihedral_type_map.items(), key=lambda x: x[1]):
            K, d, n = dihedral_type_params[key_core]
            ti, tj, tk, tl = dihedral_type_repr[key_core]
            d_int = int(round(d))
            n_int = int(round(n))
            lines.append(
                f"{dtype:5d}  {K: .3f}  {d_int:2d}  {n_int:d}  # {ti} -- {tj} -- {tk} -- {tl}"
            )
        lines.append("")

    # ---- Improper Coeffs ----
    if include_impropers and nimpropertypes > 0:
        lines.append("# Improper Coeffs (K [kcal/mol], C0, C1, C2)  ; improper_style fourier")
        lines.append("Improper Coeffs")
        lines.append("")
        for key_core, itype in sorted(improper_type_map.items(), key=lambda x: x[1]):
            K, C0, C1, C2 = improper_type_params[key_core]
            tj, ti, tk, tl = improper_type_repr[key_core]
            lines.append(
                f"{itype:5d}  {K: .3f}  {C0: .3f}  {C1: .3f}  {C2: .3f}  "
                f"# center={tj}, others={ti},{tk},{tl}"
            )
        lines.append("")

    # ---- Determine coordinated water molecules for naming ----
    adjacency2 = build_adjacency_from_bonds(natoms, bonds)
    water_mol_index = {}

    current_w = 0
    for i in range(1, natoms + 1):
        el = species[i - 1]
        tlabel = chem_atom_types[i - 1]
        if el != "O":
            continue
        if "Owc" not in tlabel:
            continue

        neighs = adjacency2[i]
        h_neighs = [j for j in neighs if species[j - 1] == "H" and "Hwc" in chem_atom_types[j - 1]]
        if not h_neighs:
            continue

        current_w += 1
        water_mol_index[i] = current_w
        for h in h_neighs:
            water_mol_index[h] = current_w

    # ---- Atoms ----
    lines.append("Atoms # full")
    lines.append("")

    if charges is None:
        charges = [0.0] * natoms
    elif len(charges) < natoms:
        charges = list(charges) + [0.0] * (natoms - len(charges))

    for i, site in enumerate(structure.sites, start=1):
        lbl = atom_lmp_labels[i - 1]
        atype = type_id[lbl]
        q = charges[i - 1]

        el = species[i - 1]
        x, y, z = site.coords
        molid = 1

        if i in water_mol_index:
            widx = water_mol_index[i]
            if el == "O":
                atom_name = f"Owc_{widx}"
            else:
                atom_name = f"Hwc_{widx}"
        else:
            atom_name = lbl

        lines.append(
            f"{i:5d}  {molid:3d}  {atype:3d}  {q: .8f}  {x: .6f}  {y: .6f}  {z: .6f}  "
            f"# {el} {atom_name} (type={lbl})"
        )
    lines.append("")

    # ---- Bonds ----
    lines.append("Bonds")
    lines.append("")
    for idx, (i, j, dist) in enumerate(bonds, start=1):
        ti = atom_lmp_labels[i - 1]
        tj = atom_lmp_labels[j - 1]
        key = tuple(sorted((ti, tj)))
        btype = bond_type_map[key]
        lines.append(f"{idx:5d}  {btype:3d}  {i:5d}  {j:5d}  # {ti} -- {tj}")
    lines.append("")

    # ---- Angles ----
    lines.append("Angles")
    lines.append("")
    for idx, (i, j, k, key) in enumerate(angle_instances, start=1):
        atype = angle_type_map[key]
        ti = atom_lmp_labels[i - 1]
        tj = atom_lmp_labels[j - 1]
        tk = atom_lmp_labels[k - 1]
        lines.append(f"{idx:5d}  {atype:3d}  {i:5d}  {j:5d}  {k:5d}  # {ti} -- {tj} -- {tk}")
    lines.append("")

    # ---- Dihedrals ----
    if include_dihedrals and ndihedrals > 0:
        lines.append("Dihedrals")
        lines.append("")
        for idx, (i, j, k, l, dtype) in enumerate(dihedral_instances, start=1):
            ti = atom_lmp_labels[i - 1]
            tj = atom_lmp_labels[j - 1]
            tk = atom_lmp_labels[k - 1]
            tl = atom_lmp_labels[l - 1]
            lines.append(
                f"{idx:5d}  {dtype:3d}  {i:5d}  {j:5d}  {k:5d}  {l:5d}  # {ti} -- {tj} -- {tk} -- {tl}"
            )
        lines.append("")

    # ---- Impropers ----
    if include_impropers and nimpropers > 0:
        lines.append("Impropers")
        lines.append("")
        for idx, (j, i, k, l, itype) in enumerate(improper_instances, start=1):
            tj = atom_lmp_labels[j - 1]
            ti = atom_lmp_labels[i - 1]
            tk = atom_lmp_labels[k - 1]
            tl = atom_lmp_labels[l - 1]
            lines.append(
                f"{idx:5d}  {itype:3d}  {j:5d}  {i:5d}  {k:5d}  {l:5d}  # center={tj}, others={ti},{tk},{tl}"
            )
        lines.append("")

    return "\n".join(lines)


# -------------------- GUI Application --------------------


class TopologyBuilderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CIF → LAMMPS Topology Builder (UFF + restrained UFF + DDEC6 + QEq + SPC/Fw + free protons)")
        self.geometry("1500x900")

        self.cif_path = tk.StringVar()
        self.uff_path = tk.StringVar()
        self.ddec_path = tk.StringVar()
        self.out_path = tk.StringVar()

        self.covalent_scale = tk.DoubleVar(value=1.2)
        self.tree_elements = None
        self.tree_pairs = None
        self.pair_cutoffs = {}
        self.selected_pair_var = tk.StringVar()
        self.selected_cutoff_var = tk.StringVar()

        self.atom_types = []
        self.atom_fragments = []
        self.tree_atoms = None

        self.ff_mode = tk.StringVar(value="UFF_all_cryst")
        self.water_model = tk.StringVar(value="none")

        # Charge model: DDEC or QEq
        self.charge_mode = tk.StringVar(value="qeq")

        # restrained UFF controls
        self.restrain_bond_slider = tk.DoubleVar(value=100.0)   # 0–100 %
        self.restrain_angle_slider = tk.DoubleVar(value=100.0)  # 0–100 %
        self.restrain_bonds = tk.BooleanVar(value=True)
        self.restrain_angles = tk.BooleanVar(value=True)
        # restraint policy: how k is modified when using CIF equilibrium
        # 'soften_mismatch' (safe): k decreases with |Δ|; never stiffens
        # 'recompute_cif' (UFF-consistent): recompute k using CIF r0/θ0 (can soften or stiffen)
        self.restrain_policy = tk.StringVar(value="soften_mismatch")

        # free proton indices (1-based)
        self.free_proton_str = tk.StringVar(value="")

        # toggles to include dihedral/improper sections
        self.include_dihedrals = tk.BooleanVar(value=True)
        self.include_impropers = tk.BooleanVar(value=True)

        # Replication settings
        self.enable_replication = tk.BooleanVar(value=False)
        self.rep_nx = tk.IntVar(value=1)
        self.rep_ny = tk.IntVar(value=1)
        self.rep_nz = tk.IntVar(value=1)

        self.linker_env_groups = {}

        self.text_output = None
        self.text_log = None

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        frame_input = ttk.Frame(notebook)
        frame_bonds = ttk.Frame(notebook)
        frame_types = ttk.Frame(notebook)
        frame_ff = ttk.Frame(notebook)
        frame_rep = ttk.Frame(notebook)
        frame_output = ttk.Frame(notebook)

        notebook.add(frame_input, text="Input / Options")
        notebook.add(frame_bonds, text="Bond Detection")
        notebook.add(frame_types, text="Atom Types")
        notebook.add(frame_ff, text="Force Field & Charges")
        notebook.add(frame_rep, text="Cell Replication")
        notebook.add(frame_output, text="Preview & Log")

        # ---- Input tab ----
        for col in range(3):
            frame_input.columnconfigure(col, weight=1)

        ttk.Label(
            frame_input,
            text="CIF → LAMMPS topology builder",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(10, 15))

        ttk.Label(frame_input, text="CIF file:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        entry_cif = ttk.Entry(frame_input, textvariable=self.cif_path)
        entry_cif.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame_input, text="Browse…", command=self.browse_cif).grid(
            row=1, column=2, sticky="w", padx=5, pady=5
        )

        ttk.Label(frame_input, text="Output LAMMPS data file:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        entry_out = ttk.Entry(frame_input, textvariable=self.out_path)
        entry_out.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame_input, text="Browse…", command=self.browse_out).grid(
            row=2, column=2, sticky="w", padx=5, pady=5
        )

        ttk.Label(
            frame_input,
            text="1) Use 'Bond Detection' to set bond cutoffs.\n"
                 "2) Optionally run 'Atom Types' to inspect types + SBU/LINKER/GUEST/COUNTERION flags.\n"
                 "3) Configure force field/charges and replication, then generate data in 'Preview & Log'.",
            foreground="#555",
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5))

        # ---- Bond Detection tab ----
        for col in range(2):
            frame_bonds.columnconfigure(col, weight=1)
        frame_bonds.rowconfigure(1, weight=1)

        ttk.Label(
            frame_bonds,
            text="Bond detection: elements and pairwise cutoffs",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(10, 10), padx=5, sticky="w")

        elem_frame = ttk.LabelFrame(frame_bonds, text="Elements in CIF")
        elem_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        elem_frame.rowconfigure(0, weight=1)
        elem_frame.columnconfigure(0, weight=1)

        self.tree_elements = ttk.Treeview(
            elem_frame, columns=("Element", "Count"), show="headings", height=10
        )
        self.tree_elements.heading("Element", text="Element")
        self.tree_elements.heading("Count", text="Count")
        self.tree_elements.column("Element", width=80, anchor="center")
        self.tree_elements.column("Count", width=80, anchor="center")
        scroll_elem = ttk.Scrollbar(elem_frame, orient="vertical", command=self.tree_elements.yview)
        self.tree_elements.configure(yscrollcommand=scroll_elem.set)
        self.tree_elements.grid(row=0, column=0, sticky="nsew")
        scroll_elem.grid(row=0, column=1, sticky="ns")

        pair_frame = ttk.LabelFrame(frame_bonds, text="Possible pairs & cutoffs")
        pair_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        pair_frame.rowconfigure(0, weight=1)
        pair_frame.columnconfigure(0, weight=1)

        self.tree_pairs = ttk.Treeview(
            pair_frame, columns=("Pair", "Cutoff"), show="headings", height=10
        )
        self.tree_pairs.heading("Pair", text="Pair (A-B)")
        self.tree_pairs.heading("Cutoff", text="Cutoff [Å]")
        self.tree_pairs.column("Pair", width=120, anchor="center")
        self.tree_pairs.column("Cutoff", width=100, anchor="center")
        scroll_pair = ttk.Scrollbar(pair_frame, orient="vertical", command=self.tree_pairs.yview)
        self.tree_pairs.configure(yscrollcommand=scroll_pair.set)
        self.tree_pairs.grid(row=0, column=0, sticky="nsew")
        scroll_pair.grid(row=0, column=1, sticky="ns")
        self.tree_pairs.bind("<<TreeviewSelect>>", self.on_pair_select)

        control_frame = ttk.Frame(frame_bonds)
        control_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 10))
        for c in range(7):
            control_frame.columnconfigure(c, weight=0)
        control_frame.columnconfigure(0, weight=1)

        ttk.Label(control_frame, text="Global covalent radius scale:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Spinbox(
            control_frame,
            from_=1.0,
            to=2.0,
            increment=0.05,
            textvariable=self.covalent_scale,
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Scan CIF for elements & pairs",
            command=self.scan_bond_pairs,
        ).grid(row=0, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(
            control_frame,
            text="Default: d_ij ≤ scale · (r_cov(i) + r_cov(j)). You can override per-pair and save/load settings.",
            foreground="#555",
        ).grid(row=1, column=0, columnspan=7, sticky="w", padx=5, pady=(0, 5))

        ttk.Label(control_frame, text="Selected pair:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.selected_pair_var, state="readonly", width=12).grid(
            row=2, column=1, sticky="w", padx=5, pady=5
        )

        ttk.Label(control_frame, text="Cutoff [Å]:").grid(row=2, column=2, sticky="e", padx=5, pady=5)
        ttk.Entry(control_frame, textvariable=self.selected_cutoff_var, width=10).grid(
            row=2, column=3, sticky="w", padx=5, pady=5
        )

        ttk.Button(
            control_frame,
            text="Apply to selected",
            command=self.apply_cutoff_to_selected,
        ).grid(row=2, column=4, sticky="w", padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Save bond settings…",
            command=self.save_bond_settings,
        ).grid(row=2, column=5, sticky="w", padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Load bond settings…",
            command=self.load_bond_settings,
        ).grid(row=2, column=6, sticky="w", padx=5, pady=5)

        # ---- Atom Types tab ----
        for col in range(1):
            frame_types.columnconfigure(col, weight=1)
        frame_types.rowconfigure(1, weight=1)

        ttk.Label(
            frame_types,
            text="Atom types & fragments (SBU / LINKER / GUEST / COUNTERION)",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, pady=(10, 5), padx=5, sticky="w")

        ttk.Button(
            frame_types,
            text="Analyze atom types",
            command=self.analyze_atom_types_gui,
        ).grid(row=0, column=0, sticky="e", padx=5, pady=5)

        frame_atoms = ttk.LabelFrame(frame_types, text="Per-atom types and fragments")
        frame_atoms.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        frame_atoms.rowconfigure(0, weight=1)
        frame_atoms.columnconfigure(0, weight=1)

        self.tree_atoms = ttk.Treeview(
            frame_atoms,
            columns=("Index", "Element", "Type", "Fragment"),
            show="headings",
            height=10,
        )
        self.tree_atoms.heading("Index", text="Idx")
        self.tree_atoms.heading("Element", text="El")
        self.tree_atoms.heading("Type", text="Type")
        self.tree_atoms.heading("Fragment", text="Frag")
        self.tree_atoms.column("Index", width=40, anchor="center")
        self.tree_atoms.column("Element", width=40, anchor="center")
        self.tree_atoms.column("Type", width=260, anchor="w")
        self.tree_atoms.column("Fragment", width=120, anchor="center")
        scroll_atoms = ttk.Scrollbar(frame_atoms, orient="vertical", command=self.tree_atoms.yview)
        self.tree_atoms.configure(yscrollcommand=scroll_atoms.set)
        self.tree_atoms.grid(row=0, column=0, sticky="nsew")
        scroll_atoms.grid(row=0, column=1, sticky="ns")

        # ---- Force field & charges tab ----
        for col in range(3):
            frame_ff.columnconfigure(col, weight=1)

        ttk.Label(
            frame_ff,
            text="Force field & charges",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(10, 15))

        ttk.Label(frame_ff, text="Force field mode:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ff_frame = ttk.Frame(frame_ff)
        ff_frame.grid(row=1, column=1, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(
            ff_frame,
            text="Restrained UFF: UFF k + crystallographic r0/θ0",
            variable=self.ff_mode,
            value="UFF_all_cryst",
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            ff_frame,
            text="Pure UFF: UFF k + UFF r0/θ0",
            variable=self.ff_mode,
            value="UFF_all_uff",
        ).grid(row=1, column=0, sticky="w")

        ttk.Label(frame_ff, text="Force field file (UFF; Python or table):").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(frame_ff, textvariable=self.uff_path).grid(
            row=2, column=1, sticky="ew", padx=5, pady=5
        )
        ttk.Button(frame_ff, text="Browse…", command=self.browse_uff).grid(
            row=2, column=2, sticky="w", padx=5, pady=5
        )

        ttk.Label(
            frame_ff,
            text="Python file must define UFF_DATA = {type: (r1, theta0, x1, D1, zeta, Z1, Vi, Uj, Xi, Hard, Radius), ...}.",
            foreground="#555",
        ).grid(row=3, column=1, columnspan=2, sticky="w", padx=5, pady=(0, 10))

        ttk.Label(frame_ff, text="DDEC6 charges XYZ (for DDEC option):").grid(
            row=4, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(frame_ff, textvariable=self.ddec_path).grid(
            row=4, column=1, sticky="ew", padx=5, pady=5
        )
        ttk.Button(frame_ff, text="Browse…", command=self.browse_ddec).grid(
            row=4, column=2, sticky="w", padx=5, pady=5
        )

        ttk.Label(
            frame_ff,
            text="If using SPC/Fw and no DDEC file, QEq charges will be corrected after adding SPC/Fw water charges.",
            foreground="#555",
        ).grid(row=5, column=1, columnspan=2, sticky="w", padx=5, pady=(0, 15))

        # Charge model selection
        charge_frame = ttk.LabelFrame(frame_ff, text="Charge model")
        charge_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=(5, 5))
        charge_frame.columnconfigure(0, weight=1)

        ttk.Radiobutton(
            charge_frame,
            text="DDEC6 charges (from XYZ file)",
            variable=self.charge_mode,
            value="ddec",
        ).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Radiobutton(
            charge_frame,
            text="QEq from UFF (χ, η)",
            variable=self.charge_mode,
            value="qeq",
        ).grid(row=1, column=0, sticky="w", padx=5, pady=2)

        ttk.Label(frame_ff, text="Water model:").grid(row=7, column=0, sticky="e", padx=5, pady=5)
        water_frame = ttk.Frame(frame_ff)
        water_frame.grid(row=7, column=1, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(
            water_frame,
            text="No special model",
            variable=self.water_model,
            value="none",
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            water_frame,
            text="SPC/Fw for water (Ow/Hw/Owc/Hwc)",
            variable=self.water_model,
            value="SPC/Fw",
        ).grid(row=1, column=0, sticky="w")

        # restrained UFF controls
        restrain_frame = ttk.LabelFrame(frame_ff, text="Restrained UFF controls")
        restrain_frame.grid(row=8, column=0, columnspan=3, sticky="ew", padx=5, pady=(5, 5))
        restrain_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            restrain_frame,
            text="Optimise / restrain BONDS (UFF → CIF)",
            variable=self.restrain_bonds,
        ).grid(row=0, column=0, sticky="w", padx=5, pady=3)

        bond_scale = ttk.Scale(
            restrain_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.restrain_bond_slider,
        )
        bond_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=3)

        ttk.Entry(
            restrain_frame,
            textvariable=self.restrain_bond_slider,
            width=5,
            state="readonly",
        ).grid(row=0, column=2, sticky="w", padx=5, pady=3)

        ttk.Checkbutton(
            restrain_frame,
            text="Optimise / restrain ANGLES (UFF → CIF)",
            variable=self.restrain_angles,
        ).grid(row=1, column=0, sticky="w", padx=5, pady=3)

        angle_scale = ttk.Scale(
            restrain_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.restrain_angle_slider,
        )
        angle_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=3)

        ttk.Entry(
            restrain_frame,
            textvariable=self.restrain_angle_slider,
            width=5,
            state="readonly",
        ).grid(row=1, column=2, sticky="w", padx=5, pady=3)

        ttk.Label(
            restrain_frame,
            text="restrain policy:",
            foreground="#555",
        ).grid(row=2, column=0, sticky="w", padx=5, pady=(0, 3))

        policy_cb = ttk.Combobox(
            restrain_frame,
            textvariable=self.restrain_policy,
            state="readonly",
            values=[
                "soften_mismatch",
                "recompute_cif",
            ],
            width=28,
        )
        policy_cb.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=(0, 3))

        ttk.Label(
            restrain_frame,
            text="Slider: 0 → pure UFF; 100 → use CIF equilibrium. Policy controls whether k is only softened (soften_mismatch) or recomputed at CIF geometry (recompute_cif).",
            foreground="#555",
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 3))
        # topology simplifications
        rm_frame = ttk.LabelFrame(frame_ff, text="Optional topology simplifications")
        rm_frame.grid(row=9, column=0, columnspan=3, sticky="ew", padx=5, pady=(5, 5))
        ttk.Checkbutton(
            rm_frame,
            text="Include Dihedrals section",
            variable=self.include_dihedrals,
        ).grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(
            rm_frame,
            text="Include Impropers section",
            variable=self.include_impropers,
        ).grid(row=1, column=0, sticky="w", padx=5, pady=3)

        # Free proton method
        fp_frame = ttk.LabelFrame(frame_ff, text="Free proton method (optional)")
        fp_frame.grid(row=10, column=0, columnspan=3, sticky="ew", padx=5, pady=(5, 10))

        ttk.Label(
            fp_frame,
            text="H atom indices to free (1-based, comma/space separated):",
        ).grid(row=0, column=0, sticky="w", padx=5, pady=3)

        ttk.Entry(fp_frame, textvariable=self.free_proton_str, width=40).grid(
            row=0, column=1, sticky="w", padx=5, pady=3
        )

        ttk.Label(
            fp_frame,
            text=(
                "Selected H will lose all bonds/angles/dihedrals/impropers\n"
                "and use LJ σ = 1.238 Å, ε = 1·10⁻⁶ kcal/mol. Charges are unchanged."
            ),
            foreground="#555",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 3))

        # ---- Cell Replication tab ----
        for col in range(2):
            frame_rep.columnconfigure(col, weight=1)

        ttk.Label(
            frame_rep,
            text="Supercell replication (Nx × Ny × Nz)",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(10, 10), padx=5, sticky="w")

        ttk.Checkbutton(
            frame_rep,
            text="Enable replication of the unit cell before building topology",
            variable=self.enable_replication,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        rep_grid = ttk.Frame(frame_rep)
        rep_grid.grid(row=2, column=0, columnspan=2, sticky="w", padx=20, pady=5)

        ttk.Label(rep_grid, text="Nx:").grid(row=0, column=0, sticky="e", padx=5, pady=3)
        ttk.Spinbox(rep_grid, from_=1, to=20, textvariable=self.rep_nx, width=5).grid(
            row=0, column=1, sticky="w", padx=5, pady=3
        )

        ttk.Label(rep_grid, text="Ny:").grid(row=0, column=2, sticky="e", padx=5, pady=3)
        ttk.Spinbox(rep_grid, from_=1, to=20, textvariable=self.rep_ny, width=5).grid(
            row=0, column=3, sticky="w", padx=5, pady=3
        )

        ttk.Label(rep_grid, text="Nz:").grid(row=0, column=4, sticky="e", padx=5, pady=3)
        ttk.Spinbox(rep_grid, from_=1, to=20, textvariable=self.rep_nz, width=5).grid(
            row=0, column=5, sticky="w", padx=5, pady=3
        )

        ttk.Label(
            frame_rep,
            text=(
                "When enabled, the CIF unit cell is first expanded into an Nx×Ny×Nz supercell.\n"
                "Bonds, angles, dihedrals and impropers are then built on the supercell, so connectivity\n"
                "across former cell boundaries is fully explicit in the LAMMPS topology."
            ),
            foreground="#555",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 10))

        # ---- Preview & Log tab ----
        frame_output.rowconfigure(0, weight=1)
        frame_output.columnconfigure(0, weight=3)
        frame_output.columnconfigure(1, weight=2)

        self.text_output = tk.Text(frame_output, wrap="none", font=("Consolas", 9))
        self.text_log = tk.Text(frame_output, wrap="word", font=("Segoe UI", 9))

        scroll_y_out = ttk.Scrollbar(frame_output, orient="vertical", command=self.text_output.yview)
        scroll_y_log = ttk.Scrollbar(frame_output, orient="vertical", command=self.text_log.yview)

        self.text_output.configure(yscrollcommand=scroll_y_out.set)
        self.text_log.configure(yscrollcommand=scroll_y_log.set)

        self.text_output.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=5)
        scroll_y_out.grid(row=0, column=0, sticky="nse", padx=(0, 5), pady=5)

        self.text_log.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        scroll_y_log.grid(row=0, column=1, sticky="nse", padx=(0, 5), pady=5)

        bottom_bar = ttk.Frame(frame_output)
        bottom_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        bottom_bar.columnconfigure(0, weight=1)

        ttk.Button(
            bottom_bar,
            text="Generate LAMMPS data",
            command=self.run_generation,
        ).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Button(
            bottom_bar,
            text="Save current data to file…",
            command=self.save_output_from_preview,
        ).grid(row=0, column=1, sticky="e", padx=5)

    # ---- File browsers ----
    def browse_cif(self):
        path = filedialog.askopenfilename(
            title="Choose CIF file",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")],
        )
        if path:
            self.cif_path.set(path)

    def browse_uff(self):
        path = filedialog.askopenfilename(
            title="Choose UFF file",
            filetypes=[
                ("Python FF files", "*.py"),
                ("Text FF files", "*.ff *.txt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.uff_path.set(path)

    def browse_ddec(self):
        path = filedialog.askopenfilename(
            title="Choose DDEC6 XYZ file",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
        )
        if path:
            self.ddec_path.set(path)

    def browse_out(self):
        path = filedialog.asksaveasfilename(
            title="Save LAMMPS data file as",
            defaultextension=".data",
            filetypes=[("LAMMPS data", "*.data *.lmp"), ("All files", "*.*")],
        )
        if path:
            self.out_path.set(path)

    # ---- Bond settings ----
    def scan_bond_pairs(self):
        self.log("Scanning CIF for elements & possible pairs...")
        cif = self.cif_path.get().strip()
        if not cif:
            messagebox.showerror("Error", "Please select a CIF file first.")
            return
        try:
            structure = Structure.from_file(cif)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CIF:\n{e}")
            self.log("ERROR reading CIF for bond detection:")
            self.log(traceback.format_exc())
            return

        species = [site.specie.symbol for site in structure.sites]
        counts = Counter(species)
        unique_elems = sorted(counts.keys())

        for tree in (self.tree_elements, self.tree_pairs):
            if tree is None:
                continue
            for item in tree.get_children():
                tree.delete(item)

        self.selected_pair_var.set("")
        self.selected_cutoff_var.set("")
        self.pair_cutoffs = {}

        for el in unique_elems:
            self.tree_elements.insert("", "end", values=(el, counts[el]))

        cov_r = {}
        for el in unique_elems:
            try:
                r = Element(el).covalent_radius
            except Exception:
                r = None
            if r is None:
                r = 0.7
            cov_r[el] = r

        cov_scale = float(self.covalent_scale.get())
        pair_count = 0
        for a, b in itertools.combinations_with_replacement(unique_elems, 2):
            r0 = cov_r[a] + cov_r[b]
            cutoff = cov_scale * r0
            key = tuple(sorted((a, b)))
            self.pair_cutoffs[key] = cutoff
            pair_label = f"{a}-{b}"
            self.tree_pairs.insert("", "end", values=(pair_label, f"{cutoff:.3f}"))
            pair_count += 1

        self.log(f"Found {len(unique_elems)} unique elements and {pair_count} possible pairs.")

    def on_pair_select(self, event=None):
        sel = self.tree_pairs.selection()
        if not sel:
            return
        item_id = sel[0]
        values = self.tree_pairs.item(item_id, "values")
        if not values:
            return
        self.selected_pair_var.set(values[0])
        self.selected_cutoff_var.set(values[1])

    def apply_cutoff_to_selected(self):
        pair_label = self.selected_pair_var.get().strip()
        if not pair_label:
            messagebox.showinfo("No pair selected", "Please select a pair in the table first.")
            return
        try:
            new_cutoff = float(self.selected_cutoff_var.get())
        except ValueError:
            messagebox.showerror("Invalid cutoff", "Please enter a valid numeric cutoff (in Å).")
            return

        try:
            a, b = pair_label.split("-")
        except ValueError:
            messagebox.showerror("Invalid pair label", f"Cannot parse pair label: {pair_label}")
            return

        key = tuple(sorted((a, b)))
        self.pair_cutoffs[key] = new_cutoff

        for iid in self.tree_pairs.get_children():
            vals = self.tree_pairs.item(iid, "values")
            if vals and vals[0] == pair_label:
                self.tree_pairs.item(iid, values=(pair_label, f"{new_cutoff:.3f}"))
                break

        self.log(f"Updated cutoff for pair {pair_label} to {new_cutoff:.3f} Å.")

    def save_bond_settings(self):
        if not self.pair_cutoffs:
            messagebox.showinfo("No settings", "No pair cutoffs to save. Scan CIF first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save bond detection settings",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        data = {
            "covalent_scale": float(self.covalent_scale.get()),
            "pair_cutoffs": {f"{a}-{b}": float(c) for (a, b), c in self.pair_cutoffs.items()},
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.log(f"Saved bond detection settings to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save bond detection settings:\n{e}")
            self.log(f"ERROR saving bond detection settings: {e}")

    def load_bond_settings(self):
        path = filedialog.askopenfilename(
            title="Load bond detection settings",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load bond detection settings:\n{e}")
            self.log(f"ERROR loading bond detection settings: {e}")
            return

        cov_scale = data.get("covalent_scale", 1.2)
        self.covalent_scale.set(cov_scale)

        raw_cutoffs = data.get("pair_cutoffs", {})
        new_pair_cutoffs = {}
        for key_str, cut in raw_cutoffs.items():
            try:
                a, b = key_str.split("-")
            except ValueError:
                continue
            new_pair_cutoffs[tuple(sorted((a, b)))] = float(cut)
        self.pair_cutoffs = new_pair_cutoffs

        if self.tree_pairs is not None:
            for item in self.tree_pairs.get_children():
                self.tree_pairs.delete(item)
            for (a, b), cut in sorted(self.pair_cutoffs.items()):
                pair_label = f"{a}-{b}"
                self.tree_pairs.insert("", "end", values=(pair_label, f"{cut:.3f}"))

        self.log(f"Loaded bond detection settings from {path}")

    # ---- Atom typing ----
    def analyze_atom_types_gui(self):
        cif = self.cif_path.get().strip()
        if not cif:
            messagebox.showerror("Error", "Please select a CIF file in 'Input / Options'.")
            return

        self.log("Analyzing atom types and fragments...")

        try:
            structure = Structure.from_file(cif)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CIF:\n{e}")
            self.log("ERROR reading CIF for atom typing:")
            self.log(traceback.format_exc())
            return

        cov_scale = float(self.covalent_scale.get())
        pair_cutoffs = self.pair_cutoffs if self.pair_cutoffs else None

        try:
            bonds, angles, dihedrals, impropers = build_topology(
                structure,
                pair_cutoffs=pair_cutoffs,
                covalent_scale=cov_scale,
            )
        except Exception:
            self.log("ERROR building topology (for atom typing):")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", "Failed to build topology. See log.")
            return

        self.log(f"(Typing) Found {len(bonds)} bonds for the current settings.")

        atom_types, fragments = assign_atom_types_and_fragments(structure, bonds)
        self.atom_types = atom_types
        self.atom_fragments = fragments

        for item in self.tree_atoms.get_children():
            self.tree_atoms.delete(item)

        species = [site.specie.symbol for site in structure.sites]
        for idx in range(1, len(structure) + 1):
            self.tree_atoms.insert(
                "",
                "end",
                values=(idx, species[idx - 1], atom_types[idx - 1], fragments[idx - 1]),
            )

        _, linker_groups = analyze_linker_environments(structure, bonds, atom_types)
        self.linker_env_groups = linker_groups

        self.log("Atom typing & fragment analysis completed.")

    # ---- Logging ----
    def log(self, msg):
        if self.text_log is not None:
            self.text_log.insert("end", msg + "\n")
            self.text_log.see("end")
            self.update_idletasks()

    # ---- Generation ----
    def run_generation(self):
        self.text_log.delete("1.0", "end")
        self.text_output.delete("1.0", "end")

        cif = self.cif_path.get().strip()
        if not cif:
            messagebox.showerror("Error", "Please select a CIF file.")
            return

        cov_scale = float(self.covalent_scale.get())
        self.log(f"Using CIF: {cif}")
        self.log(f"Global covalent radius scale: {cov_scale:.3f}")

        # --- Read unit-cell structure ---
        try:
            structure_unit = Structure.from_file(cif)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CIF:\n{e}")
            self.log("ERROR reading CIF:")
            self.log(traceback.format_exc())
            return

        self.log(f"Unit cell has {len(structure_unit)} atoms.")
        species_unit = [site.specie.symbol for site in structure_unit.sites]

        # --- Parse free proton indices on UNIT CELL and tag them as a site property ---
        n_unit = len(structure_unit)
        free_mask_unit = [False] * n_unit

        fp_str = self.free_proton_str.get().strip()
        if fp_str:
            tokens = re.split(r"[,\s]+", fp_str)
            for tok in tokens:
                if not tok:
                    continue
                try:
                    idx = int(tok)
                except ValueError:
                    self.log(f"WARNING: cannot parse free proton index '{tok}', skipping.")
                    continue

                if idx < 1 or idx > n_unit:
                    self.log(f"WARNING: free proton index {idx} out of UNIT CELL range (1..{n_unit}), skipping.")
                    continue

                if species_unit[idx - 1] != "H":
                    self.log(f"WARNING: unit-cell atom {idx} is {species_unit[idx - 1]}, not H; skipping.")
                    continue

                free_mask_unit[idx - 1] = True

        # Attach mask so replication keeps it aligned to sites
        try:
            structure_unit.add_site_property("free_proton", list(free_mask_unit))
        except Exception:
            try:
                structure_unit.remove_site_property("free_proton")
            except Exception:
                pass
            structure_unit.add_site_property("free_proton", list(free_mask_unit))

        # --- Load UFF parameters ---
        uff_path = self.uff_path.get().strip()
        if uff_path:
            self.log(f"Loading UFF parameters from: {uff_path}")
            try:
                uff = UFFParameters(uff_path)
            except Exception:
                self.log("ERROR loading UFF parameters, continuing with simple defaults.")
                self.log(traceback.format_exc())
                uff = UFFParameters(None)
        else:
            self.log("No UFF file provided, using simple defaults (QEq will not work properly).")
            uff = UFFParameters(None)

        # --- Charges: compute on UNIT CELL only, then replicate ---
        ddec_path = self.ddec_path.get().strip()
        charge_mode = self.charge_mode.get()
        charges_unit = None

        if charge_mode == "ddec":
            if ddec_path:
                self.log(f"Loading DDEC6 charges from: {ddec_path}")
                try:
                    ddec_atoms = load_ddec6_xyz(ddec_path)
                    charges_unit = map_charges_to_structure(structure_unit, ddec_atoms)
                    self.log(f"Assigned charges to {len(charges_unit)} atoms from DDEC6 (unit cell).")
                except Exception:
                    self.log("ERROR loading DDEC6 charges, continuing with zero charges.")
                    self.log(traceback.format_exc())
                    charges_unit = None
            else:
                self.log("Charge mode 'DDEC6' selected but no DDEC file given → charges will start from 0.0.")
                charges_unit = None

        elif charge_mode == "qeq":
            if uff is None or not isinstance(uff, UFFParameters) or not uff.type_params:
                self.log("Charge mode 'QEq' selected but UFF parameters are missing/empty → charges will be 0.0.")
                charges_unit = None
            else:
                try:
                    self.log("Computing QEq charges from UFF electronegativities and hardness (unit cell)...")
                    charges_unit = compute_qeq_charges(structure_unit, uff, total_charge=0.0)
                    self.log(f"QEq: computed charges for {len(charges_unit)} atoms (unit cell).")
                    qsum = sum(charges_unit)
                    self.log(f"QEq: total charge (unit cell) = {qsum:+.4f} (target = 0.0000 before SPC/Fw corrections).")
                except Exception:
                    self.log("ERROR while computing QEq charges, falling back to zero charges.")
                    self.log(traceback.format_exc())
                    charges_unit = None

        if charges_unit is None:
            charges_unit = [0.0] * len(structure_unit)

        # Attach charges as a site property so replication keeps mapping
        try:
            structure_unit.add_site_property("charge", list(charges_unit))
        except Exception:
            try:
                structure_unit.remove_site_property("charge")
            except Exception:
                pass
            structure_unit.add_site_property("charge", list(charges_unit))

        # --- Replication settings ---
        if self.enable_replication.get():
            nx = max(1, int(self.rep_nx.get()))
            ny = max(1, int(self.rep_ny.get()))
            nz = max(1, int(self.rep_nz.get()))
        else:
            nx = ny = nz = 1

        if nx * ny * nz == 1:
            self.log("Replication: disabled (using 1×1×1 unit cell).")
            structure = structure_unit
        else:
            self.log(f"Replication: building supercell {nx} × {ny} × {nz} ...")
            structure = structure_unit.copy()
            try:
                structure.make_supercell([nx, ny, nz])
            except Exception:
                self.log("ERROR during make_supercell; falling back to unit cell.")
                self.log(traceback.format_exc())
                structure = structure_unit
                nx = ny = nz = 1
            self.log(f"Supercell has {len(structure)} atoms.")

        # Extract replicated charges from site properties
        if "charge" in structure.site_properties:
            charges = list(structure.site_properties["charge"])
        else:
            charges = [0.0] * len(structure)

        # Species of final structure
        species = [site.specie.symbol for site in structure.sites]

        # --- Build free proton indices from replicated site property ---
        free_proton_indices = set()
        if "free_proton" in structure.site_properties:
            mask = structure.site_properties["free_proton"]
            for i, flag in enumerate(mask, start=1):  # 1-based
                if flag:
                    free_proton_indices.add(i)

        if free_proton_indices:
            self.log(
                "Free proton method: requested free H indices (replication-safe) = "
                + ", ".join(str(i) for i in sorted(free_proton_indices))
            )

        # --- Bond detection settings ---
        pair_cutoffs = self.pair_cutoffs if self.pair_cutoffs else None
        if pair_cutoffs:
            self.log("Using custom per-pair bond detection cutoffs from Bond Detection tab.")
        else:
            self.log("No per-pair cutoffs defined; using global covalent radius scale for all pairs.")

        # --- Build topology on final structure ---
        self.log("Building topology (bonds, angles, dihedrals, impropers)...")
        try:
            bonds, angles, dihedrals, impropers = build_topology(
                structure,
                pair_cutoffs=pair_cutoffs,
                covalent_scale=cov_scale,
            )
        except Exception:
            self.log("ERROR building topology:")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", "Failed to build topology. See log.")
            return

        self.log(
            f"Found {len(bonds)} bonds, {len(angles)} angles, "
            f"{len(dihedrals)} dihedrals, {len(impropers)} impropers."
        )

        # --- Atom types + fragments for this topology ---
        try:
            atom_types, fragments = assign_atom_types_and_fragments(structure, bonds)

            # Free proton: overwrite type + force fragment to GUEST
            if free_proton_indices:
                for idx in free_proton_indices:
                    if 1 <= idx <= len(atom_types):
                        atom_types[idx - 1] = "H_PROTON"  # <- your desired base name
                        fragments[idx - 1] = "GUEST"  # <- force as guest

            self.atom_types = atom_types
            self.atom_fragments = fragments

            chem_types = [f"{t}_{frag}" for t, frag in zip(atom_types, fragments)]
            type_counts = Counter(chem_types)
            self.log("Atom typing summary (top types):")
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
                self.log(f"  {t}: {c}")
        except Exception:
            self.log("WARNING: atom typing failed (non-fatal).")
            self.log(traceback.format_exc())
            chem_types = None
            fragments = None

        # --- Remove all bonds/angles/dihedrals/impropers involving free protons ---
        if free_proton_indices:
            nb_before = len(bonds)
            bonds = [
                (i, j, d)
                for (i, j, d) in bonds
                if (i not in free_proton_indices and j not in free_proton_indices)
            ]
            self.log(
                f"Free proton method: removed {nb_before - len(bonds)} bonds involving free H."
            )

            na_before = len(angles)
            angles = [
                (i, j, k)
                for (i, j, k) in angles
                if (
                    i not in free_proton_indices
                    and j not in free_proton_indices
                    and k not in free_proton_indices
                )
            ]
            self.log(
                f"Free proton method: removed {na_before - len(angles)} angles involving free H."
            )

            nd_before = len(dihedrals)
            dihedrals = [
                (i, j, k, l)
                for (i, j, k, l) in dihedrals
                if (
                    i not in free_proton_indices
                    and j not in free_proton_indices
                    and k not in free_proton_indices
                    and l not in free_proton_indices
                )
            ]
            self.log(
                f"Free proton method: removed {nd_before - len(dihedrals)} dihedrals involving free H."
            )

            ni_before = len(impropers)
            impropers = [
                (j, i, k, l)
                for (j, i, k, l) in impropers
                if (
                    j not in free_proton_indices
                    and i not in free_proton_indices
                    and k not in free_proton_indices
                    and l not in free_proton_indices
                )
            ]
            self.log(
                f"Free proton method: removed {ni_before - len(impropers)} impropers involving free H."
            )

        ff_mode = self.ff_mode.get()
        use_spcfw = (self.water_model.get() == "SPC/Fw")

        # --- Apply SPC/Fw charges on lattice water + final charge correction ---
        if use_spcfw:
            self.log("SPC/Fw water model: applying SPC/Fw LJ parameters (in writer) and charges for lattice waters.")
            spcfw_qO = -0.82
            spcfw_qH = 0.41

            nat = len(structure)
            lattice_mask = [False] * nat

            for idx in range(nat):
                t = atom_types[idx]
                if t in ("Ow", "Hw"):
                    lattice_mask[idx] = True
                    if t == "Ow":
                        charges[idx] = spcfw_qO
                    else:
                        charges[idx] = spcfw_qH

            Q_before = sum(charges)
            n_framework = sum(1 for m in lattice_mask if not m)
            if n_framework > 0:
                delta = Q_before / n_framework
                for i in range(nat):
                    if not lattice_mask[i]:
                        charges[i] -= delta
                Q_after = sum(charges)
                self.log(
                    f"Final charge correction with SPC/Fw: initial Σq = {Q_before:+.8f}, "
                    f"N_framework = {n_framework}, Δq/atom = {delta:+.8e}, "
                    f"final Σq (double) = {Q_after:+.8f}."
                )
            else:
                Q_after = sum(charges)
                self.log("Final charge correction: no framework atoms (only lattice water?) – skipping correction.")

            # NEUTRALITY FIX AT PRINTING PRECISION (8 decimals)
            q_print = [round(q, 8) for q in charges]
            total_print = sum(q_print)

            if abs(total_print) > 5e-7:
                idx_corr = None
                for i in range(nat):
                    if not lattice_mask[i]:
                        idx_corr = i
                        break

                if idx_corr is not None:
                    q_print[idx_corr] = round(q_print[idx_corr] - total_print, 8)
                    total_final = sum(q_print)
                    self.log(
                        f"Neutrality fix after rounding: Σq_print_before_fix = {total_print:+.8f}, "
                        f"Σq_print_final = {total_final:+.8f}."
                    )
                else:
                    self.log(
                        "Neutrality fix: no framework atom found to adjust after rounding "
                        "(only lattice waters?)."
                    )
            else:
                self.log(
                    f"Neutrality fix: Σq_print already ≈ 0 (Σq_print = {total_print:+.8f}), no extra adjustment."
                )

            charges = q_print
        else:
            Q_total = sum(charges)
            self.log(f"No SPC/Fw water model; total charge before writing data = {Q_total:+.8f}.")

        # --- Restraint factors ---
        if ff_mode == "UFF_all_cryst":
            if self.restrain_bonds.get():
                s_bond = max(0.0, min(1.0, self.restrain_bond_slider.get() / 100.0))
            else:
                s_bond = 0.0
            if self.restrain_angles.get():
                s_angle = max(0.0, min(1.0, self.restrain_angle_slider.get() / 100.0))
            else:
                s_angle = 0.0
            self.log(f"Restrained UFF: bond restraint = {s_bond*100:.1f}%, angle restraint = {s_angle*100:.1f}%, policy = {self.restrain_policy.get()}.")
        else:
            s_bond = 0.0
            s_angle = 0.0
            self.log("Force-field mode is 'Pure UFF' – restrained UFF sliders are inactive.")

        # --- Build LAMMPS data text ---
        try:
            data_str = write_lammps_data(
                structure,
                bonds,
                angles,
                dihedrals,
                impropers,
                uff=uff,
                charges=charges,
                chem_atom_types=chem_types,
                fragments=fragments,
                ff_mode=ff_mode,
                use_spcfw=use_spcfw,
                include_dihedrals=self.include_dihedrals.get(),
                include_impropers=self.include_impropers.get(),
                restrain_bond_factor=s_bond,
                restrain_angle_factor=s_angle,
                restrain_policy=self.restrain_policy.get(),
                title=os.path.basename(cif),
            )
        except Exception:
            self.log("ERROR writing LAMMPS data:")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", "Failed to build LAMMPS data text. See log.")
            return

        self.text_output.insert("1.0", data_str)
        self.text_output.see("1.0")
        self.log("LAMMPS data built. You can save it from the 'Preview & Log' tab.")

        out = self.out_path.get().strip()
        if out:
            try:
                with open(out, "w") as f:
                    f.write(data_str)
                self.log(f"Saved LAMMPS data to: {out}")
            except Exception as e:
                self.log(f"ERROR saving LAMMPS data to {out}: {e}")

    def save_output_from_preview(self):
        data_text = self.text_output.get("1.0", "end").strip()
        if not data_text:
            messagebox.showinfo("Nothing to save", "No data in preview.")
            return
        path = filedialog.asksaveasfilename(
            title="Save LAMMPS data file as",
            defaultextension=".data",
            filetypes=[("LAMMPS data", "*.data *.lmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(data_text)
            self.log(f"Saved LAMMPS data to: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data:\n{e}")
            self.log(f"ERROR saving data: {e}")


if __name__ == "__main__":
    app = TopologyBuilderApp()
    app.mainloop()
