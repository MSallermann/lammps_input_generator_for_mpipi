from pathlib import Path

from mpipi_lammps_gen.generate_lammps_files import (
    add_proteins_to_lammps_data,
    get_lammps_group_definition,
    initialize_lammps_data,
    write_lammps_data_file,
)
from mpipi_lammps_gen.globular_domains import GlobularDomain
from mpipi_lammps_gen.util import sequence_to_prot_data_spiral


def test_mixtures():
    # Set up two proteins
    p1 = sequence_to_prot_data_spiral("PR" * 25)
    domains1 = [GlobularDomain([(2, 8), (11, 16)])]

    p2 = sequence_to_prot_data_spiral("A" * 35)
    domains2 = [GlobularDomain([(30, 32)])]

    lammps_data = initialize_lammps_data()
    add_proteins_to_lammps_data(
        lammps_data, p1, domains1, n_proteins_x=2, n_proteins_y=2, n_proteins_z=2
    )

    add_proteins_to_lammps_data(
        lammps_data,
        p2,
        domains2,
        n_proteins_x=2,
        n_proteins_y=2,
        n_proteins_z=1,
        offset_z=lammps_data.z_lims[1] - lammps_data.z_lims[0] + 5,
    )

    grps = get_lammps_group_definition(lammps_data)

    with Path("mixtures.data").open("w") as f:
        f.write(write_lammps_data_file(lammps_data))

    with Path("grps_mixtures.txt").open("w") as f:
        f.write(grps)
