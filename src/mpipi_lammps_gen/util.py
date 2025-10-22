from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.spatial import distance

from mpipi_lammps_gen.generate_lammps_files import GlobularDomain


def group_distance_matrix(
    residue_positions: Sequence[tuple[float, float, float]],
    g1: GlobularDomain,
    g2: GlobularDomain,
) -> npt.NDArray:
    idx_start_1 = g1.start_idx()
    idx_end_1 = g1.end_idx()
    group1_coords = residue_positions[idx_start_1:idx_end_1]

    idx_start_2 = g2.start_idx()
    idx_end_2 = g2.end_idx()
    group2_coords = residue_positions[idx_start_2:idx_end_2]

    return distance.cdist(group1_coords, group2_coords)


def coordination_numbers_from_distance_matrix(
    distance_matrix: npt.NDArray, cutoff: float
) -> tuple[npt.NDArray, npt.NDArray]:
    within_cutooff = distance_matrix < cutoff

    coord_numbers_a = np.sum(within_cutooff, axis=1)
    coord_numbers_b = np.sum(within_cutooff, axis=0)

    return (coord_numbers_a, coord_numbers_b)
