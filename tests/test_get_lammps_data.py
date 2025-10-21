from pathlib import Path

from mpipi_lammps_gen.generate_lammps_files import (
    generate_lammps_data,
    get_lammps_group_definition,
    parse_cif_from_path,
    write_lammps_data_file,
)
from mpipi_lammps_gen.globular_domains import decide_globular_domains_from_sequence

CIF = Path(__file__).parent / "res" / "Q9ULK0.cif"


def test_get_protein_data():
    prot_data = parse_cif_from_path(CIF)

    n_res = 580
    assert len(prot_data.plddts) == n_res
    assert len(prot_data.sequence_one_letter) == n_res
    assert len(prot_data.sequence_three_letter) == n_res
    assert len(prot_data.atom_xyz) == n_res

    threshold = 90.0
    minimum_idr_length = 16
    minimum_domain_length = 4

    globular_domains = decide_globular_domains_from_sequence(
        prot_data.plddts,
        threshold=threshold,
        minimum_domain_length=minimum_domain_length,
        minimum_idr_length=minimum_idr_length,
    )

    # NOTE: this value may change if the criterion is adjusted in the future.
    # For now we check it to guard against unexpected changes
    assert len(globular_domains) == 6

    lammps_data = generate_lammps_data(
        prot_data=prot_data, globular_domains=globular_domains, box_buffer=20.0
    )

    assert len(lammps_data.atoms) == n_res
    assert len(lammps_data.groups) == len(globular_domains)

    # hard to assert anything about these, but they should at least not crash
    write_lammps_data_file(lammps_data)
    get_lammps_group_definition(lammps_data)
