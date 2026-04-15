from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mpipi_lammps_gen.generate_lammps_files import (
    concatenate_proteins,
    generate_lammps_data,
    rigid_transform_protein,
    write_lammps_data_file,
)
from mpipi_lammps_gen.util import sequence_to_prot_data_spiral


def test_protein_data():
    p1 = sequence_to_prot_data_spiral("AR" * 25)

    # rotate 90 degrees around x axis
    rotation_matrix = Rotation.from_rotvec([90.0, 0, 0], degrees=True).as_matrix()
    translation_vector = np.array([0, 0, 5])

    print(translation_vector.shape)

    p2 = rigid_transform_protein(
        p1, rotation_matrix=rotation_matrix, translation_vector=translation_vector
    )

    p_total = concatenate_proteins(p1, p2, connecting_vector=[3.81, 0, 0])

    lammps_data = generate_lammps_data(p_total, globular_domains=[])
    data_file = write_lammps_data_file(lammps_data)

    with Path("data_file.data").open("w") as f:
        f.write(data_file)
