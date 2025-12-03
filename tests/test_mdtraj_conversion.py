from pathlib import Path

import mdtraj
import numpy as np

from mpipi_lammps_gen.generate_lammps_files import parse_cif_from_path
from mpipi_lammps_gen.mdtraj_conversion import (
    mdtraj_to_protein,
    plddt_to_bfactor,
    protein_to_mdtraj,
)

CIF = Path(__file__).parent / "res" / "Q9ULK0.cif"


def test_mdtraj():
    prot_data = parse_cif_from_path(CIF)

    traj = protein_to_mdtraj(prot_data)
    traj.save_cif("./out.cif", bfactors=plddt_to_bfactor(prot_data))
    traj.save_pdb("./out.pdb", bfactors=plddt_to_bfactor(prot_data))

    traj_from_file = mdtraj.load("./out.pdb")
    # print(traj_from_file._bfactors)

    prot_data_from_file = mdtraj_to_protein(traj_from_file)
    prot_data_from_file.compute_residue_positions()

    print(prot_data.plddts)
    print(prot_data_from_file.plddts)

    assert np.all(
        np.isclose(prot_data.residue_positions, prot_data_from_file.residue_positions)
    )

    assert all(
        s_file == s
        for s, s_file in zip(
            prot_data.sequence_one_letter,
            prot_data_from_file.sequence_one_letter,
            strict=True,
        )
    )


test_mdtraj()
