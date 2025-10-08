from mpipi_lammps_gen import generate_lammps_files
from pathlib import Path


def main():

    cif_path = Path("./example.cif")

    protein_data = generate_lammps_files.parse_cif_from_path(cif_path)

    globular_domains = generate_lammps_files.decide_globular_domains(
        protein_data.plddts
    )

    lammps_data = generate_lammps_files.generate_lammps_data(
        protein_data, globular_domains
    )

    data_file_str = generate_lammps_files.write_lammps_data_file(lammps_data)

    with open("lammps.data", "w") as f:
        f.write(data_file_str)

    print(generate_lammps_files.get_lammps_group_script(lammps_data))


if __name__ == "__main__":
    main()
