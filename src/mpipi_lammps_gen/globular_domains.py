from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


@dataclass
class GlobularDomain:
    indices: list[tuple[int, int]]

    def start_idx(self) -> int:
        if len(self.indices) == 0:
            return -1

        return self.indices[0][0]

    def end_idx(self) -> int:
        if len(self.indices) == 0:
            return -1
        return self.indices[-1][1]

    def size_total(self) -> int:
        return self.end_idx() - self.start_idx()

    def n_atoms(self) -> int:
        return sum(t[1] - t[0] for t in self.indices)

    def is_in_rigid_region(self, idx: int) -> bool:
        return any(idx >= t[0] and idx <= t[1] for t in self.indices)

    def to_lammps_indices(self) -> list[tuple[int, int]]:
        """Returns the indices with a +1 added, since LAMMPS counts from 1."""
        return [(t[0] + 1, t[1] + 1) for t in self.indices]

    def get_all_indices(self) -> list[int]:
        """Returns a flat list containing all the indices of residues in the group"""
        indices = []
        for t in self.indices:
            indices.extend(range(t[0], t[1] + 1))
        return indices

    @staticmethod
    def merge(g1: GlobularDomain, g2: GlobularDomain) -> GlobularDomain:
        # Make sure thet the indices keep monotonically increasing
        if g1.end_idx() < g2.start_idx():
            return GlobularDomain(indices=g1.indices + g2.indices)

        return GlobularDomain(indices=g2.indices + g1.indices)

    @staticmethod
    def fuse(g1: GlobularDomain, g2: GlobularDomain) -> GlobularDomain:
        # Make sure thet the indices keep monotonically increasing
        if g2.end_idx() < g1.start_idx():
            return GlobularDomain.fuse(g2, g1)

        new_indices = g1.indices[:-1]

        if len(g1.indices) == 0:
            new_indices.extend(g2.indices)
        elif len(g2.indices) == 0:
            new_indices.extend(g1.indices)
        else:
            new_indices.append((g1.indices[-1][0], g2.indices[0][1]))
            new_indices.extend(g2.indices[1:])

        return GlobularDomain(new_indices)


def decide_globular_domains_from_sequence(
    plddts: Iterable[float],
    threshold: float = 70.0,
    minimum_domain_length: int = 3,
    minimum_idr_length: int = 3,
    fuse: bool = True,
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
    if len(in_globular_domain) > 0 and in_globular_domain[0]:
        start_indices.insert(0, 0)

    if len(in_globular_domain) > 0 and in_globular_domain[-1]:
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

            if fuse:
                globular_domains[idx + 1] = GlobularDomain.fuse(domain1, domain2)
            else:
                globular_domains[idx + 1] = GlobularDomain.merge(domain1, domain2)

    # iterate in reverse order, because then the lower indices do not change due to the `pop`
    for i in sorted(indices_to_remove, reverse=True):
        globular_domains.pop(i)

    return globular_domains


def merge_domains(
    domains: Sequence[GlobularDomain],
    should_be_merged: Callable[[GlobularDomain, GlobularDomain], bool],
    fuse: bool = False,
) -> list[GlobularDomain]:
    n_domains = len(domains)

    # first we create an undicrected graph of everything that has to be merged
    merge_graph = nx.Graph()

    # every domain is a node
    merge_graph.add_nodes_from(range(n_domains))

    # add an edge for every time should_be_merged is true
    for i1, g1 in enumerate(domains):
        for j, g2 in enumerate(
            domains[i1 + 1 :]
        ):  # we start from i1+1 to save up on unnecessary work and "skip self interaction"
            i2 = i1 + 1 + j  # we have to remember this offset here
            if should_be_merged(g1, g2):
                merge_graph.add_edge(i1, i2)

    # Now we have to find out the connected components
    components = nx.connected_components(merge_graph)

    new_domains = []
    for comp in components:
        g = GlobularDomain([])
        for i in comp:
            if fuse:
                g = GlobularDomain.fuse(g, domains[i])
            else:
                g = GlobularDomain.merge(g, domains[i])
        new_domains.append(g)

    return new_domains


def build_protein_graph(n_residues: int, domains: Sequence[GlobularDomain]) -> nx.Graph:
    graph = nx.Graph()

    # add all the groups as nodes
    for i, _ in enumerate(domains):
        graph.add_node(f"CD{i}")

    def belongs_to_group(idx_residue: int) -> int | None:
        for i, d in enumerate(domains):
            if d.is_in_rigid_region(idx_residue):
                return i

        return None

    for i_res in range(n_residues):
        # if the residue does not belong to any of the domains we add a node
        this_grp_idx = belongs_to_group(i_res)

        if this_grp_idx is None:
            this_node = f"Res{i_res}"
            graph.add_node(this_node)
        else:
            this_node = f"CD{this_grp_idx}"

        # now decide which edges to add
        if i_res > 0:
            i_res_prev = i_res - 1
            prev_grp_idx = belongs_to_group(i_res_prev)

            if prev_grp_idx is None:
                prev_node = f"Res{i_res_prev}"
            else:
                prev_node = f"CD{prev_grp_idx}"

            graph.add_edge(prev_node, this_node)

    return graph


def shortest_path_matrix(
    n_residues: int, domains: Sequence[GlobularDomain], protein_graph: nx.Graph
):
    distance_matrix = np.zeros((n_residues, n_residues))

    def get_grp_node(idx_residue: int) -> str | None:
        for i, d in enumerate(domains):
            if d.is_in_rigid_region(idx_residue):
                return f"CD{i}"

        return None

    def get_node(i_res: int) -> str:
        grp_node = get_grp_node(i_res)
        if grp_node is not None:
            return grp_node
        return f"Res{i_res}"

    for i in range(n_residues):
        node_i = get_node(i)
        for j in range(i + 1, n_residues):
            node_j = get_node(j)

            path_len = nx.shortest_path_length(protein_graph, node_i, node_j)

            distance_matrix[i, j] = path_len
            distance_matrix[j, i] = path_len

    return distance_matrix


def protein_topology(
    n_residues: int,
    domains: Sequence[GlobularDomain],
) -> nx.Graph:
    graph = nx.Graph()

    # add all the groups as nodes
    for i, d in enumerate(domains):
        graph.add_node(f"CD{i}", weight=d.n_atoms())

    # add a start and an end node
    graph.add_node("start", weight=1)
    graph.add_node("end", weight=1)

    def get_grp_idx(idx_residue: int) -> int | None:
        for i, d in enumerate(domains):
            if d.is_in_rigid_region(idx_residue):
                return i

        return None

    # starting node of the current edge
    edge_start = "start"
    edge_start_idx = 0
    grp_idx_prev_res = None

    last_proper_group = "start"

    for i_res in range(n_residues):
        this_grp_idx = get_grp_idx(i_res)

        # only add an edge if the previous node was not in this group
        if this_grp_idx is not None and this_grp_idx != grp_idx_prev_res:
            graph.add_edge(
                edge_start,
                f"CD{this_grp_idx}",
                length=i_res - edge_start_idx,
                weight=1.0 / (i_res - edge_start_idx + 1),
                loop=(edge_start == f"CD{this_grp_idx}"),
            )
            last_proper_group = f"CD{this_grp_idx}"
        elif (
            this_grp_idx is None and grp_idx_prev_res is not None
        ):  # leaving a group, start counting for an edge again
            edge_start = f"CD{grp_idx_prev_res}"
            edge_start_idx = i_res - 1

        grp_idx_prev_res = this_grp_idx

    graph.add_edge(last_proper_group, "end", length=n_residues - 1 - edge_start_idx)

    return graph


def protein_topology2(
    n_residues: int,
    domains: Sequence[GlobularDomain],
    # residue_positions: list[tuple[float, float, float]],
) -> nx.Graph:
    graph = nx.Graph()

    # add a start and an end node
    graph.add_node("start", weight=1)
    graph.add_node("end", weight=1)

    # add all the groups as nodes
    for i, d in enumerate(domains):
        graph.add_node(f"CD{i}", weight=d.n_atoms(), indices=set(d.get_all_indices()))

    def get_grp_idx(idx_residue: int) -> int | None:
        for i, d in enumerate(domains):
            if d.is_in_rigid_region(idx_residue):
                return i

        return None

    # starting node of the current edge
    edge_start = "start"
    edge_start_idx = 0

    grp_idx_prev_res = None

    last_proper_group = "start"

    for i_res in range(n_residues):
        this_grp_idx = get_grp_idx(i_res)

        if this_grp_idx is not None and this_grp_idx != grp_idx_prev_res:
            # we end the current edge
            graph.add_edge(
                edge_start,
                f"CD{this_grp_idx}",
                length=i_res - edge_start_idx,
                weight=1.0 / (i_res - edge_start_idx + 1),
                start_idx=edge_start_idx,
                end_idx=i_res,
                loop=(edge_start == f"CD{this_grp_idx}"),
            )
            last_proper_group = f"CD{this_grp_idx}"
        elif (
            this_grp_idx is None and grp_idx_prev_res is not None
        ):  # leaving a group, start counting for an edge again
            edge_start = f"CD{grp_idx_prev_res}"
            edge_start_idx = i_res - 1

        grp_idx_prev_res = this_grp_idx

    graph.add_edge(
        last_proper_group,
        "end",
        length=n_residues - 1 - edge_start_idx,
        start_idx=edge_start_idx,
        end_idx=n_residues - 1,
        loop=False,
    )

    return graph
