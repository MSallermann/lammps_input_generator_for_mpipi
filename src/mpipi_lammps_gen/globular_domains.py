from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class GlobularDomain:
    indices: list[tuple[int, int]]

    def start_idx(self) -> int:
        return self.indices[0][0]

    def end_idx(self) -> int:
        return self.indices[-1][1]

    def size_total(self) -> int:
        return self.end_idx() - self.start_idx()

    def is_in_rigid_region(self, idx: int) -> bool:
        return any(idx >= t[0] and idx <= t[1] for t in self.indices)

    def to_lammps_indices(self) -> list[tuple[int, int]]:
        """Returns the indices with a +1 added, since LAMMPS counts from 1."""
        return [(t[0] + 1, t[1] + 1) for t in self.indices]


def decide_globular_domains_from_sequence(
    plddts: Iterable[float],
    threshold: float = 70.0,
    minimum_domain_length: int = 3,
    minimum_idr_length: int = 3,
) -> list[GlobularDomain]:
    in_globular_domain = [p > threshold for p in plddts]
    n_res = len(in_globular_domain)

    # Indices where value changes from False to True
    start_indices = [
        i
        for i in range(1, n_res)
        if not in_globular_domain[i - 1] and in_globular_domain[i]
    ]

    # Indices where value changes from True to False
    end_indices = [
        i
        for i in range(n_res - 1)
        if in_globular_domain[i] and not in_globular_domain[i + 1]
    ]

    # Special checks for the first and last index, since we cannot detect them based on changes from False to True
    if in_globular_domain[0]:
        start_indices.insert(0, 0)

    if in_globular_domain[-1]:
        end_indices.append(n_res - 1)

    assert len(start_indices) == len(end_indices)

    globular_domains = [
        GlobularDomain(indices=[(s, e)])
        for s, e in zip(start_indices, end_indices, strict=True)
    ]

    # remove domains, which are below the minimum globular domain length
    indices_to_remove = []

    for idx, domain in enumerate(globular_domains):
        for pair in domain.indices:
            if pair[1] - pair[0] + 1 < minimum_domain_length:
                indices_to_remove.append(idx)  # noqa: PERF401

    for i in sorted(indices_to_remove, reverse=True):
        globular_domains.pop(i)

    # remove IDRs, which are below the minimum IDR length by "merging" domains
    indices_to_remove = []

    for idx in range(len(globular_domains) - 1):
        domain1 = globular_domains[idx]
        last_idx1 = domain1.indices[-1][1]

        domain2 = globular_domains[idx + 1]
        first_idx2 = domain2.indices[0][0]

        if first_idx2 - last_idx1 - 1 < minimum_idr_length:
            indices_to_remove.append(
                idx
            )  # We remove domain 1 and merge it into domain2
            # we just set both pairs to the new "merged" value
            globular_domains[idx + 1].indices = list(domain1.indices + domain2.indices)

    # iterate in reverse order, because then the lower indices do not change due to the `pop`
    for i in sorted(indices_to_remove, reverse=True):
        globular_domains.pop(i)

    return globular_domains
