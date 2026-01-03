LAMForge

LAMForge automatically converts crystal structures into ready-to-run LAMMPS data files by constructing topology, assigning chemistry-aware force fields and charges, and handling proton release and supercell replication. It is designed for complex inorganic and hybrid systems such as MOFs and metal phosphonates.

🔁 How it Works

LAMForge follows a deterministic, reproducible workflow:

1. Read a crystal structure (P1 cif file)
2. Detect bonding topology using geometry and chemical rules
3. Assign chemistry-aware atom types and structural fragments
4. Build bonded interactions (bonds, angles, dihedrals, impropers)
5. Assign force-field parameters (UFF-based)
6. Assign partial charges (DDEC6 or QEq)
7. Assign water model (SPC/FW)
8. Handle proton release and mobile species
9. Replicate the structure if requested
10. Write a complete, LAMMPS-compatible data file

## ▶ How to Run

LAMForge requires Python 3.9 or newer.

1. Clone the repository and open a terminal in the project directory
2. Install dependencies: py -m pip install -r requirements.txt
3. Run the script: py lamforge.py

📖 Citation

If you use LAMForge in academic work, please cite:

K. Xanthopoulos, LAMForge: Automatic topology and force-field generator for LAMMPS,
GitHub repository, https://github.com/xanthop-chem/LAMForge

🙏 Acknowledgements

LAMForge was inspired by existing tools for preparing LAMMPS simulations, in particular the **LAMMPS Interface** workflow.  
All code was written independently and no code from other projects is included.
The UFF parameters where taken from the peteboyd/lammps_interface repository.

Parts of the design, debugging, and documentation benefited from interactive assistance provided by **ChatGPT (OpenAI)**.  
Final implementation, validation, and scientific decisions are solely the responsibility of the author.

📜 License

MIT License

🚧 Status

Active research software.
The API and features may evolve.
