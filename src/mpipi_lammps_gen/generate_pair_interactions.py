from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class WFInteraction:
    epsilon: float
    sigma: float
    mu: int
    nu: int
    rc: float


__default_interactions_df__ = pl.read_csv(
    Path(__file__).parent / "default_interactions.csv"
)


class PairDict(dict):
    def __getitem__(self, key: tuple[int, int]) -> Any:
        if key[1] > key[0]:
            return super().__getitem__(key)

        return super().__getitem__((key[1], key[0]))

    def __setitem__(self, key: Any, value: Any) -> None:
        if key[1] > key[0]:
            return super().__setitem__(key, value)

        return super().__setitem__((key[1], key[0]), value)


__default_interactions__ = PairDict(
    {
        (row["i"] - 1, row["j"] - 1): WFInteraction(
            epsilon=row["epsilon"],
            sigma=row["sigma"],
            mu=row["mu"],
            nu=row["nu"],
            rc=row["rc"],
        )
        for row in __default_interactions_df__.iter_rows(named=True)
    }
)


def generate_wf_interactions(
    idr_glob_scaling: float, glob_glob_scaling: float, n_res: int = 20
) -> PairDict:
    interactions = PairDict()

    # There are n_res residues, which means we have to iterate up to 2*n_res, to have a variatn of each residue which is also in a globular domain
    for i in range(2 * n_res):
        for j in range(i, 2 * n_res):
            # Check if i and j are in an intrinsically disoredered region (IDR)
            i_is_in_idr = i <= n_res
            j_is_in_idr = j <= n_res

            # If not, we also need to grab the corresponding indices that these residue types would have in an IDR
            i_idr = i if i_is_in_idr else i - n_res
            j_idr = j if j_is_in_idr else j - n_res

            # Look up the default IDR-IDR WF parameters for this interaction

            idr_to_idr_interaction = __default_interactions__[(i_idr, j_idr)]
            # Scale the epsilon based on IDR-IDR (1.0), IDR-GLOB, GLOB-GLOB prefactors
            if i_is_in_idr and j_is_in_idr:
                epsilon = idr_to_idr_interaction.epsilon
            elif i_is_in_idr or j_is_in_idr:
                epsilon = idr_glob_scaling * idr_to_idr_interaction.epsilon
            else:
                epsilon = glob_glob_scaling * idr_to_idr_interaction.epsilon

            # put the new interaction
            interactions[(i, j)] = WFInteraction(
                epsilon=epsilon,
                sigma=idr_to_idr_interaction.sigma,
                mu=idr_to_idr_interaction.mu,
                nu=idr_to_idr_interaction.nu,
                rc=idr_to_idr_interaction.rc,
            )

    return interactions


def get_wf_pairs_str(interactions: PairDict) -> str:
    res = ""

    base = "pair_coeff {i} {j} wf/cut {epsilon} {sigma} {nu} {mu} {rc}\n"

    for k, v in interactions.items():
        i, j = k
        res += base.format(i=i + 1, j=j + 1, **asdict(v))

    return res
