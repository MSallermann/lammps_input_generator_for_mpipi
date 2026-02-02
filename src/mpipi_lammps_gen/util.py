import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.spatial import distance

from mpipi_lammps_gen.generate_lammps_files import (
    GlobularDomain,
    ProteinData,
    is_valid_one_letter_sequence,
)


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


def sequence_to_prot_data_spiral(
    seq: str,
    n_res_per_ring: int = 16,
    radius_spiral: float = 10.0,
    distance_z: float = 10.0,
) -> ProteinData:
    assert is_valid_one_letter_sequence(seq)

    res_positions = [
        (
            radius_spiral * math.sin(idx_res / n_res_per_ring * 2.0 * math.pi),
            radius_spiral * math.cos(idx_res / n_res_per_ring * 2.0 * math.pi),
            distance_z / n_res_per_ring * idx_res,
        )
        for idx_res in range(len(seq))
    ]

    plddts = [0.0] * len(seq)

    return ProteinData(
        atom_xyz=None,
        atom_types=None,
        residue_positions=res_positions,
        sequence_one_letter=list(seq),
        sequence_three_letter=None,
        pae=None,
        plddts=plddts,
    )


def read_polars(inp: Path | str) -> pl.DataFrame:
    inp = Path(inp)
    if inp.suffix.lower() == ".parquet":
        return pl.read_parquet(inp)
    if inp.suffix.lower() in [".feather", ".arrow"]:
        return pl.read_ipc(inp)
    if inp.suffix.lower() == ".csv":
        return pl.read_csv(inp)
    msg = f"Dont know suffix {inp.suffix}"
    raise Exception(msg)


def to_fasta(sequences: Iterable[str], names: Iterable[str] | None = None) -> str:
    sequence_list = list(sequences)

    if names is None:
        names = [f"sequence{i}" for i in range(len(sequence_list))]

    res = ""
    for seq, name in zip(sequence_list, names, strict=True):
        res += f">{name}\n{seq.upper()}\n"

    return res
