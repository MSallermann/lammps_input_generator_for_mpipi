from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


@dataclass
class GlobularDomain:
    """Represents a globular domain in a protein."""

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


def protein_topology(  # noqa: PLR0912
    n_residues: int,
    domains: Sequence[GlobularDomain],
    chain_break_after: Iterable[int] | None = None,
) -> nx.MultiGraph:
    """
    Generate an undirected NetworkX MultiGraph representing protein topology.

    Compared with the original version, this supports multiple terminal pairs.
    A chain break after residue i means there is no backbone continuity between
    residue i and residue i + 1.

    For the old single-chain case:
        terminal nodes are still named "start" and "end".

    For multi-segment cases:
        terminal nodes are named "start_0", "end_0", "start_1", "end_1", ...

    Globular domain nodes are still shared globally as "CD{i}".
    """

    graph = nx.MultiGraph()

    chain_break_after_set = set(chain_break_after or [])

    for idx in chain_break_after_set:
        if idx < 0 or idx >= n_residues - 1:
            msg = f"Invalid chain break after residue {idx}. "
            f"Expected 0 <= idx < {n_residues - 1}."
            raise ValueError(msg)

    # Add globular domain nodes.
    for i, d in enumerate(domains):
        graph.add_node(
            f"CD{i}",
            weight=d.n_atoms(),
            indices=set(d.get_all_indices()),
            kind="domain",
        )

    # Build residue -> domain lookup.
    residue_to_domain: list[int | None] = [None] * n_residues

    for i, d in enumerate(domains):
        for idx in d.get_all_indices():
            if idx < 0 or idx >= n_residues:
                msg = f"Domain {i} contains residue index {idx},"
                f"but n_residues={n_residues}."
                raise ValueError(msg)

            if residue_to_domain[idx] is not None:
                msg = (
                    f"Residue {idx} belongs to multiple domains: "
                    f"{residue_to_domain[idx]} and {i}"
                )
                raise ValueError(msg)

            residue_to_domain[idx] = i

    # Convert break positions into continuous residue intervals.
    #
    # Example:
    #   n_residues = 10
    #   chain_break_after = {3, 6}
    #
    # gives:
    #   (0, 3), (4, 6), (7, 9)
    segment_starts = [0]
    segment_ends = []

    for break_after in sorted(chain_break_after_set):
        segment_ends.append(break_after)
        segment_starts.append(break_after + 1)

    segment_ends.append(n_residues - 1)

    segments = list(zip(segment_starts, segment_ends, strict=True))

    single_segment = len(segments) == 1

    def terminal_names(segment_idx: int) -> tuple[str, str]:
        if single_segment:
            return "start", "end"
        return f"start_{segment_idx}", f"end_{segment_idx}"

    for segment_idx, (seg_start, seg_end) in enumerate(segments):
        start_node, end_node = terminal_names(segment_idx)

        graph.add_node(
            start_node,
            weight=1,
            indices={seg_start},
            kind="terminus",
            terminus="start",
            segment_idx=segment_idx,
        )
        graph.add_node(
            end_node,
            weight=1,
            indices={seg_end},
            kind="terminus",
            terminus="end",
            segment_idx=segment_idx,
        )

        edge_start = start_node
        edge_start_idx = seg_start
        grp_idx_prev_res = None

        for i_res in range(seg_start, seg_end + 1):
            this_grp_idx = residue_to_domain[i_res]

            # Entering a domain from an IDR or from the segment start.
            if this_grp_idx is not None and this_grp_idx != grp_idx_prev_res:
                end_idx = i_res - 1

                if end_idx >= edge_start_idx:
                    segment_length = end_idx - edge_start_idx + 1
                    target_node = f"CD{this_grp_idx}"

                    graph.add_edge(
                        edge_start,
                        target_node,
                        length=segment_length,
                        weight=1.0 / segment_length,
                        start_idx=edge_start_idx,
                        end_idx=end_idx,
                        loop=(edge_start == target_node),
                        kind="idr",
                        segment_idx=segment_idx,
                    )

            # Leaving a domain into an IDR.
            elif this_grp_idx is None and grp_idx_prev_res is not None:
                edge_start = f"CD{grp_idx_prev_res}"
                edge_start_idx = i_res

            grp_idx_prev_res = this_grp_idx

        # Add final IDR tail to this segment's terminal end.
        if grp_idx_prev_res is None:
            final_length = seg_end - edge_start_idx + 1

            if final_length > 0:
                graph.add_edge(
                    edge_start,
                    end_node,
                    length=final_length,
                    weight=1.0 / final_length,
                    start_idx=edge_start_idx,
                    end_idx=seg_end,
                    loop=False,
                    kind="idr",
                    segment_idx=segment_idx,
                )

    return graph


def _find_node_of_residue(graph: nx.Graph, i1: int) -> str | None:
    for n, data in graph.nodes(data=True):
        if "indices" in data and i1 in data["indices"]:
            return n
    return None


def _find_edge_of_residue(graph: nx.Graph, i1: int) -> tuple[str, str] | None:
    for source, target, data in graph.edges(data=True):
        if data["start_idx"] <= i1 and data["end_idx"] >= i1:
            return (source, target)
    return None


def _find_connected_indices(graph: nx.Graph, n1: str, n2: str) -> tuple[int, int]:
    """Find the indices."""

    neighbours = graph[n1]

    if n2 not in neighbours:
        msg = f"{n1} and {n2} do not seem to be connected by an edge."
        raise Exception(msg)

    edge = neighbours[n2]

    edge_start_idx = edge["start_idx"]
    edge_end_idx = edge["end_idx"]

    n1_indices = graph.nodes[n1]["indices"]
    n2_indices = graph.nodes[n2]["indices"]

    if edge_start_idx in n1_indices and edge_end_idx in n2_indices:
        idx1 = edge_start_idx
        idx2 = edge_end_idx
    elif edge_start_idx in n2_indices and edge_end_idx in n1_indices:
        idx1 = edge_end_idx
        idx2 = edge_start_idx
    else:
        msg = "Could not find edge start/end indices in nodes."
        raise Exception(msg)

    return (idx1, idx2)


@dataclass
class PathProperties:
    """The properties of a path connecting two residues in a 'ProteinTopology' graph."""

    path: list[str]  # The path of nodes
    n_random_segments: int  # The number of random segments
    n_random_segments_offset: int

    # List of fixed distances. These correspond to "shortcuts" through globular domains
    fixed_distances: list[float]

    # A list of loop tuples. The first index is the index in the loop and the last index is the length of the loop ().
    # A path generally may start and/or end in a looped segment
    loops: list[tuple[int, int]]

    n1: str | None
    n2: str | None
    e1: tuple[str, str] | None
    e2: tuple[str, str] | None
    node_path_start: str
    node_path_end: str


def get_path_properties(  # noqa: PLR0912, PLR0915
    graph: nx.Graph,
    i1: int,
    i2: int,
    _residue_positions: list[tuple[float, float, float]],
) -> PathProperties:
    residue_positions = np.asarray(_residue_positions, dtype=float)  # type: ignore

    # Quickly check some indices
    if i1 < 0 or i2 < 0:
        msg = "Indices must be larger than zero"
        raise Exception(msg)

    if i1 >= len(residue_positions) or i2 >= len(residue_positions):
        msg = f"Indices must be smaller than the number of residues ({len(residue_positions)})"
        raise Exception(msg)

    # this offset is necessary because some residues might lie in the "middle" of an edge
    # ... since we count the entire edge, from source to target, when we loop over the shortest paths later
    # ... we record an offset to correct for this
    n_random_segments_offset: int = 0

    # A shortest path might originate or end in a looped IDR
    # ... in this list we record (idx_within_loop, length_of_loop) tuples
    loops: list[tuple[int, int]] = []

    # first we check if the initial/final residues lie within a node
    # ... this happens if they lie within a globular domain
    n1 = _find_node_of_residue(graph, i1)
    n2 = _find_node_of_residue(graph, i2)

    # if not, we check if they are within an edge
    # ... this happens if they lie within an IDR
    e1 = None
    is_loop1 = False
    if n1 is None:
        e1 = _find_edge_of_residue(graph, i1)

        if e1 is None:
            msg = f"Could not find an edge or a node to which residue {i1} belongs"
            raise Exception(msg)

        # we have to make sure that we get the source and target of
        # the edge in the same order they occur in the protein
        # ... this is not guaranteed by the (src,target) tuple returned by `find_edge_of_residue`
        idx1, idx2 = _find_connected_indices(graph=graph, n1=e1[0], n2=e1[1])
        node_path_start = e1[0] if idx1 < idx2 else e1[1]

        # we should only count the offset if the edge is not a loop,
        # because if it is a loop it will not count towards n_random_segments anyways
        is_loop1 = graph.edges[e1]["loop"]

        if not is_loop1:
            n_random_segments_offset += graph.edges[e1]["start_idx"] - i1
        else:
            idx_in_loop = int(i1 - graph.edges[e1]["start_idx"])
            length_loop = int(graph.edges[e1]["length"])
            loops.append((idx_in_loop, length_loop))
    else:
        node_path_start = n1

    e2 = None
    is_loop2 = False
    if n2 is None:
        e2 = _find_edge_of_residue(graph, i2)

        if e2 is None:
            msg = f"Could not find an edge or a node to which residue {i2} belongs"
            raise Exception(msg)

        # we have to make sure that we get the source and target of
        # the edge in the same order they occur in the protein
        # ... this is not guaranteed by the (src,target) tuple returned by `find_edge_of_residue`
        idx1, idx2 = _find_connected_indices(graph=graph, n1=e2[0], n2=e2[1])

        node_path_end = e2[1] if idx1 < idx2 else e2[0]

        # we should only count the offset if the edge is not a loop,
        # because if it is a loop it will not count towarss n_random_segments anyways
        is_loop2 = graph.edges[e2]["loop"]
        if not is_loop2:
            n_random_segments_offset += i2 - graph.edges[e2]["end_idx"]
        else:
            idx_in_loop = int(i2 - graph.edges[e2]["start_idx"])
            length_loop = int(graph.edges[e2]["length"])
            loops.append((idx_in_loop, length_loop))
    else:
        node_path_end = n2

    # We find the shortest path
    path = nx.shortest_path(graph, node_path_start, node_path_end)
    n_random_segments: int = n_random_segments_offset
    fixed_distances: list[float] = []

    # Handle the starting point
    if (
        n1 is not None
    ):  # ... this means the starting residue lies within a globular domain
        # we have to add the distance from the starting residue to the point at which it leaves the domain
        if len(path) > 1:
            idx1, idx2 = _find_connected_indices(graph, path[0], path[1])
        else:
            idx1 = i2
        dist = np.linalg.norm(residue_positions[i1] - residue_positions[idx1])
        if dist > 0.0:
            fixed_distances.append(float(dist))
    elif is_loop1:
        # if we have a loop, we add the shorter of the two anchor distances
        idx1, idx2 = _find_connected_indices(graph, path[0], path[1])
        idx_anchor_1, idx_anchor_2 = _find_connected_indices(graph, path[0], path[0])
        dist1 = np.linalg.norm(
            residue_positions[idx_anchor_1] - residue_positions[idx1]
        )
        dist2 = np.linalg.norm(
            residue_positions[idx_anchor_2] - residue_positions[idx1]
        )
        fixed_distances.append(float(min(dist1, dist2)))

    # Handle the end point
    if (
        n2 is not None and len(path) > 1
    ):  # ... this means the end residue lies within a globular domain
        # we have to add the distance from the end residue to the point at which the last domain was entered
        idx1, idx2 = _find_connected_indices(graph, path[-1], path[-2])
        dist = np.linalg.norm(residue_positions[i2] - residue_positions[idx1])
        if dist > 0.0:
            fixed_distances.append(float(dist))
    elif is_loop1:
        # if we have a loop, we add the shorter of the two anchor distances
        idx1, idx2 = _find_connected_indices(graph, path[-1], path[-2])
        idx_anchor_1, idx_anchor_2 = _find_connected_indices(graph, path[-1], path[-1])
        dist1 = np.linalg.norm(
            residue_positions[idx_anchor_1] - residue_positions[idx1]
        )
        dist2 = np.linalg.norm(
            residue_positions[idx_anchor_2] - residue_positions[idx1]
        )
        fixed_distances.append(float(min(dist1, dist2)))

    for i, _ in enumerate(path):
        # This computes the length of the edge between the previous and the current node
        if i >= 1:
            idx1, idx2 = _find_connected_indices(graph, path[i], path[i - 1])
            n_random_segments += int(np.abs(idx2 - idx1))

        # for each globular domain we pass through, we record the distance from enter to exit
        if i >= 1 and i < len(path) - 1:
            enter_idx, _ = _find_connected_indices(graph, n1=path[i], n2=path[i - 1])
            exit_idx, _ = _find_connected_indices(graph, n1=path[i], n2=path[i + 1])

            dist = np.linalg.norm(
                residue_positions[exit_idx] - residue_positions[enter_idx]
            )

            fixed_distances.append(float(dist))

    return PathProperties(
        path=path,
        n_random_segments=n_random_segments,
        n_random_segments_offset=n_random_segments_offset,
        fixed_distances=fixed_distances,
        loops=loops,
        n1=n1,
        n2=n2,
        e1=e1,
        e2=e2,
        node_path_start=node_path_start,
        node_path_end=node_path_end,
    )
