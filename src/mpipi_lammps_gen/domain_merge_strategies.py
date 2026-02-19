from collections.abc import Sequence

import numpy as np

from mpipi_lammps_gen.globular_domains import GlobularDomain
from mpipi_lammps_gen.util import (
    coordination_numbers_from_distance_matrix,
    group_distance_matrix,
)


class MergePAE:
    def __init__(
        self,
        min_pae_cutoff: float | None,
        mean_pae_cutoff: float | None,
        pae_matrix_arr: np.ndarray,
    ):
        self.pae_matrix_arr = pae_matrix_arr
        self.min_pae_cutoff = min_pae_cutoff
        self.mean_pae_cutoff = mean_pae_cutoff

    def __call__(self, g1: GlobularDomain, g2: GlobularDomain) -> bool:
        sub_pae = self.pae_matrix_arr[
            g1.start_idx() : g1.end_idx(), g2.start_idx() : g2.end_idx()
        ]

        min_pae = np.min(np.ravel(sub_pae))

        if self.min_pae_cutoff is not None and min_pae < self.min_pae_cutoff:
            return True

        mean_pae = np.mean(np.ravel(sub_pae))

        return bool(
            self.mean_pae_cutoff is not None and mean_pae < self.mean_pae_cutoff
        )


class MergeDistance:
    def __init__(
        self,
        min_distance_cutoff: float | None,
        max_coordination_cutoff: float | None,
        coordination_distance_cutoff: float | None,
        residue_positions: Sequence[tuple[float, float, float]],
    ):
        self.min_distance_cutoff = min_distance_cutoff
        self.max_coordination_cutoff = max_coordination_cutoff
        self.coordination_distance_cutoff = coordination_distance_cutoff
        self.residue_positions = residue_positions

    def __call__(self, g1: GlobularDomain, g2: GlobularDomain) -> bool:
        distance_matrix = group_distance_matrix(self.residue_positions, g1, g2)

        if self.min_distance_cutoff is not None:
            min_distance = np.min(np.ravel(distance_matrix))
            if min_distance <= self.min_distance_cutoff:
                return True

        if (
            self.max_coordination_cutoff is not None
            and self.coordination_distance_cutoff is not None
        ):
            coord1, coord2 = coordination_numbers_from_distance_matrix(
                distance_matrix=distance_matrix,
                cutoff=self.coordination_distance_cutoff,
            )
            if (
                np.max(coord1) >= self.max_coordination_cutoff
                or np.max(coord2) >= self.max_coordination_cutoff
            ):
                return True

        return False
