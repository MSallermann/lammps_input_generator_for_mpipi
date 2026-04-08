from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

AnchorNode: TypeAlias = tuple[str, int]
TopologyEdgeRef: TypeAlias = tuple[str, str, int]


def _edge_anchor_nodes(
    topology: nx.MultiGraph,
    edge: TopologyEdgeRef,
) -> tuple[AnchorNode, AnchorNode]:
    """
    Return the two anchor nodes connected by a topology IDR edge.

    The returned pair is ordered left-to-right in sequence space.

    For an IDR spanning [start_idx, end_idx]:
        - the left anchor is the rigid residue with the largest index < start_idx,
          or the terminal residue of "start" if that node is an endpoint
        - the right anchor is the rigid residue with the smallest index > end_idx,
          or the terminal residue of "end" if that node is an endpoint
    """
    u, v, k = edge
    data = topology.edges[u, v, k]

    start_idx = int(data["start_idx"])
    end_idx = int(data["end_idx"])

    left_candidates: list[AnchorNode] = []
    right_candidates: list[AnchorNode] = []

    for node in {u, v}:
        indices = topology.nodes[node]["indices"]

        if node == "start":
            # "start" is always the left terminal anchor
            idx = next(iter(indices))
            left_candidates.append((node, idx))
            continue

        if node == "end":
            # "end" is always the right terminal anchor
            idx = next(iter(indices))
            right_candidates.append((node, idx))
            continue

        for idx in indices:
            if idx < start_idx:
                left_candidates.append((node, idx))
            if idx > end_idx:
                right_candidates.append((node, idx))

    if not left_candidates:
        msg = (
            f"Could not find a left anchor for topology edge {(u, v, k)} "
            f"with start_idx={start_idx}"
        )
        raise ValueError(msg)

    if not right_candidates:
        msg = (
            f"Could not find a right anchor for topology edge {(u, v, k)} "
            f"with end_idx={end_idx}"
        )
        raise ValueError(msg)

    left_anchor = max(left_candidates, key=lambda x: x[1])
    right_anchor = min(right_candidates, key=lambda x: x[1])

    return left_anchor, right_anchor


def build_shortest_path_graph(
    topology: nx.MultiGraph,
    residue_positions: Sequence[Sequence[float]],
    *,
    bond_length: float,
    segment_length: float | None = None,
) -> nx.MultiGraph:
    """
    Build a weighted anchor-level graph for shortest-path calculations.

    Nodes are anchor residues represented as:
        (topology_node_name, residue_index)

    Edges are:
        - IDR edges from the topology graph
        - rigid-domain shortcut edges between all anchors in the same domain

    Weight model:
        - IDR edge distance = edge["length"] * bond_length
        - domain shortcut distance = Euclidean distance between anchor residues

    If `segment_length` is provided, all distances are divided by it, so the
    resulting weights are expressed in effective random-walk segment counts.
    Otherwise, weights are expressed in distance units.

    NOTE:
        We explicitly connect all anchors within a domain (complete graph).
        This is redundant under Euclidean distances (triangle inequality),
        but keeps the model simple and allows use of standard NetworkX
        shortest path algorithms without custom logic.
    """
    pos = np.asarray(residue_positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        msg = "residue_positions must have shape (n_residues, 3)"
        raise ValueError(msg)

    if bond_length <= 0:
        msg = "bond_length must be positive"
        raise ValueError(msg)

    if segment_length is not None and segment_length <= 0:
        msg = "segment_length must be positive if provided"
        raise ValueError(msg)

    def normalize(distance: float) -> float:
        return distance if segment_length is None else distance / segment_length

    path_graph = nx.MultiGraph()
    anchors_by_topology_node: dict[str, set[int]] = defaultdict(set)

    # Add IDR edges
    for u, v, k, data in topology.edges(keys=True, data=True):
        a_left, a_right = _edge_anchor_nodes(topology, (u, v, k))

        for anchor in (a_left, a_right):
            topo_node, residue_idx = anchor
            path_graph.add_node(
                anchor,
                topology_node=topo_node,
                residue_idx=residue_idx,
            )
            anchors_by_topology_node[topo_node].add(residue_idx)

        idr_distance = float(data["length"]) * bond_length

        path_graph.add_edge(
            a_left,
            a_right,
            weight=normalize(idr_distance),
            distance=idr_distance,
            kind="idr",
            topology_edge=(u, v, k),
            start_idx=int(data["start_idx"]),
            end_idx=int(data["end_idx"]),
            length=int(data["length"]),
            loop=bool(data["loop"]),
        )

    # Add domain shortcut edges
    for topo_node, anchor_indices in anchors_by_topology_node.items():
        if topo_node in {"start", "end"}:
            continue

        sorted_indices = sorted(anchor_indices)
        for i, idx1 in enumerate(sorted_indices):
            for idx2 in sorted_indices[i + 1 :]:
                a1 = (topo_node, idx1)
                a2 = (topo_node, idx2)

                shortcut_distance = float(np.linalg.norm(pos[idx2] - pos[idx1]))

                path_graph.add_edge(
                    a1,
                    a2,
                    weight=normalize(shortcut_distance),
                    distance=shortcut_distance,
                    kind="domain_shortcut",
                    topology_node=topo_node,
                    idx1=idx1,
                    idx2=idx2,
                )

    path_graph.graph["bond_length"] = bond_length
    path_graph.graph["segment_length"] = segment_length
    path_graph.graph["n_residues"] = len(pos)

    return path_graph


ShortestPathEdgeRef: TypeAlias = tuple[AnchorNode, AnchorNode, int]


@dataclass
class PathProperties:
    """Properties of the shortest path between two residues."""

    # Anchor-level path through the shortest-path graph
    path: list[AnchorNode]
    edge_path: list[ShortestPathEdgeRef]

    # Total cost in the same units as the shortest-path graph edge weights
    total_weight: float

    # Endpoint offsets, in the same units as `total_weight`
    start_offset: float
    end_offset: float

    # Distances of explicit rigid-domain shortcut edges along the chosen path
    fixed_distances: list[float]

    # Sum of IDR contributions along the chosen path, in the same units as the
    # graph weights (distance units if segment_length is None, otherwise segment count)
    #   ... It can be thought of as the maximum end-to-end length of the IDR segments concatenated
    random_walk_contour_length: float

    # Where the original residues live in the topology graph
    n1: str | None
    n2: str | None
    e1: TopologyEdgeRef | None
    e2: TopologyEdgeRef | None

    # Endpoint loop metadata, only populated if the start/end residue lies in a self-loop IDR
    # Tuple format: (idx_within_loop, loop_length)
    # ... note: the loop length is given in residues, *not* in segments
    # ... to get the number of segments you have to add one
    start_loop: tuple[int, int] | None
    end_loop: tuple[int, int] | None


def _find_node_of_residue(graph: nx.MultiGraph, i: int) -> str | None:
    for n, data in graph.nodes(data=True):
        if "indices" in data and i in data["indices"]:
            return n
    return None


def _find_edge_of_residue(graph: nx.MultiGraph, i: int) -> TopologyEdgeRef | None:
    for u, v, k, data in graph.edges(keys=True, data=True):
        if int(data["start_idx"]) <= i <= int(data["end_idx"]):
            return (u, v, k)
    return None


def _choose_min_weight_edge(
    g: nx.MultiGraph,
    u: AnchorNode,
    v: AnchorNode,
) -> ShortestPathEdgeRef:
    edge_dict = g.get_edge_data(u, v)
    if edge_dict is None:
        msg = f"No edge between {u!r} and {v!r}"
        raise ValueError(msg)

    key = min(edge_dict, key=lambda k: edge_dict[k]["weight"])
    return (u, v, key)


def _anchor_candidates_for_residue(
    topology: nx.MultiGraph,
    sp_graph: nx.MultiGraph,
    residue_positions: np.ndarray,
    i: int,
    *,
    bond_length: float,
    segment_length: float | None = None,
) -> list[
    tuple[
        AnchorNode,
        float,  # offset cost in graph-weight units
        str | None,  # containing topology node
        TopologyEdgeRef | None,  # containing topology edge
        tuple[int, int] | None,  # loop metadata if residue lies in self-loop IDR
    ]
]:
    """
    Return candidate anchor nodes for a residue.

    Each tuple contains:
        (anchor_node, offset_cost, containing_node, containing_edge, loop_metadata)

    If the residue lies in a rigid node:
        - candidates are all anchors belonging to that topology node
        - offset_cost is Euclidean distance to the anchor (normalized if requested)

    If the residue lies in an IDR edge:
        - candidates are the two anchors of that IDR edge
        - offset_cost is contour distance along the IDR to the anchor
        - if the IDR is a self-loop, loop_metadata is (idx_within_loop, loop_length)
    """

    def normalize(distance: float) -> float:
        return distance if segment_length is None else distance / segment_length

    # Residue lies in a rigid node
    n = _find_node_of_residue(topology, i)
    if n is not None:
        candidates = []
        for anchor in sp_graph.nodes:
            topo_node, anchor_idx = anchor
            if topo_node != n:
                continue

            if n in {"start", "end"}:
                offset_distance = 0.0
            else:
                offset_distance = float(
                    np.linalg.norm(residue_positions[i] - residue_positions[anchor_idx])
                )

            candidates.append((anchor, normalize(offset_distance), n, None, None))

        if not candidates:
            msg = f"Residue {i} lies in node {n!r}, but no anchors were found for that node."
            raise ValueError(msg)

        return candidates

    # Residue lies in an IDR edge
    e = _find_edge_of_residue(topology, i)
    if e is None:
        msg = f"Could not locate residue {i} in the topology graph"
        raise ValueError(msg)

    a_left, a_right = _edge_anchor_nodes(topology, e)
    u, v, k = e
    data = topology.edges[u, v, k]

    start_idx = int(data["start_idx"])
    end_idx = int(data["end_idx"])
    loop = bool(data["loop"])

    # contour distance from residue i to the two anchors along the IDR
    left_offset = (i - start_idx + 1) * bond_length
    right_offset = (end_idx - i + 1) * bond_length

    loop_meta: tuple[int, int] | None = None
    if loop:
        loop_meta = (int(i - start_idx), int(data["length"]))

    return [
        (a_left, normalize(left_offset), None, e, loop_meta),
        (a_right, normalize(right_offset), None, e, loop_meta),
    ]


def _find_contour_edge_of_residue(
    graph: nx.MultiGraph,
    i: int,
) -> TopologyEdgeRef | None:
    """
    Return the topology edge corresponding to the physical contour segment
    containing residue i.

    This behaves like `_find_edge_of_residue`, but terminal residues are also
    considered part of the adjacent terminal IDR edge when appropriate.

    So:
      - interior IDR residues map to their IDR edge
      - residue 0 can map to a "start"-adjacent edge if that edge starts at 0
      - residue N-1 can map to an "end"-adjacent edge if that edge ends at N-1
    """
    edge = _find_edge_of_residue(graph, i)
    if edge is not None:
        return edge

    node = _find_node_of_residue(graph, i)
    if node not in {"start", "end"}:
        return None

    matches: list[TopologyEdgeRef] = []

    for u, v, k, data in graph.edges(keys=True, data=True):
        if (node == "start" and node in (u, v) and int(data["start_idx"]) == i) or (
            node == "end" and node in (u, v) and int(data["end_idx"]) == i
        ):
            matches.append((u, v, k))

    if len(matches) > 1:
        msg = f"Residue {i} matches multiple terminal contour edges: {matches}"
        raise ValueError(msg)

    if len(matches) == 1:
        return matches[0]

    return None


def get_path_properties(  # noqa: PLR0912, PLR0915
    topology: nx.MultiGraph,
    i1: int,
    i2: int,
    residue_positions: list[tuple[float, float, float]],
    *,
    shortest_path_graph: nx.MultiGraph | None = None,
    bond_length: float,
    segment_length: float | None = None,
) -> PathProperties:
    """
    Compute shortest-path properties between two residues using the anchor-level
    shortest-path graph.

    The path cost includes:
        - IDR contour distances: length * bond_length
        - rigid-domain shortcut distances: Euclidean distance between anchors

    If `segment_length` is provided, all costs are divided by it, so the final
    weights are expressed in effective random-walk segment counts.
    """
    pos = np.asarray(residue_positions, dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 3:
        msg = "residue_positions must have shape (n_residues, 3)"
        raise ValueError(msg)

    if i1 < 0 or i2 < 0:
        msg = "Residue indices must be non-negative"
        raise ValueError(msg)

    if i1 >= len(pos) or i2 >= len(pos):
        msg = (
            f"Residue indices must be smaller than the number of residues ({len(pos)})"
        )
        raise ValueError(msg)

    if bond_length <= 0:
        msg = "bond_length must be positive"
        raise ValueError(msg)

    if segment_length is not None and segment_length <= 0:
        msg = "segment_length must be positive if provided"
        raise ValueError(msg)

    if shortest_path_graph is None:
        sp_graph = build_shortest_path_graph(
            topology,
            pos,
            bond_length=bond_length,
            segment_length=segment_length,
        )
    else:
        assert shortest_path_graph.graph["bond_length"] == bond_length
        assert shortest_path_graph.graph["segment_length"] == segment_length
        assert shortest_path_graph.graph["n_residues"] == len(pos)
        sp_graph = shortest_path_graph

    n1 = _find_node_of_residue(topology, i1)
    n2 = _find_node_of_residue(topology, i2)

    contour_e1 = _find_contour_edge_of_residue(topology, i1)
    contour_e2 = _find_contour_edge_of_residue(topology, i2)

    # Case 1: same rigid domain
    # start/end are bookkeeping nodes, not rigid domains
    if n1 is not None and n1 == n2 and n1 not in {"start", "end"}:
        dist = float(np.linalg.norm(pos[i2] - pos[i1]))
        weight = dist if segment_length is None else dist / segment_length

        return PathProperties(
            path=[],
            edge_path=[],
            total_weight=weight,
            start_offset=0.0,
            end_offset=0.0,
            fixed_distances=[dist] if dist > 0 else [],
            random_walk_contour_length=0.0,
            n1=n1,
            n2=n2,
            e1=None,
            e2=None,
            start_loop=None,
            end_loop=None,
        )

    # Case 2: same contour edge
    # This includes terminal residues when they lie on the adjacent terminal IDR.
    if contour_e1 is not None and contour_e1 == contour_e2:
        u, v, k = contour_e1
        edge_data = topology.edges[u, v, k]

        dist = abs(i2 - i1) * bond_length
        weight = dist if segment_length is None else dist / segment_length

        start_loop = None
        end_loop = None
        if bool(edge_data["loop"]):
            start_idx = int(edge_data["start_idx"])
            loop_len = int(edge_data["length"])
            start_loop = (i1 - start_idx, loop_len)
            end_loop = (i2 - start_idx, loop_len)

        return PathProperties(
            path=[],
            edge_path=[],
            total_weight=weight,
            start_offset=0.0,
            end_offset=0.0,
            fixed_distances=[],
            random_walk_contour_length=weight,
            n1=n1,
            n2=n2,
            e1=contour_e1,
            e2=contour_e2,
            start_loop=start_loop,
            end_loop=end_loop,
        )

    # For the remaining cases, use the original node/edge classification
    e1 = None if n1 is not None else _find_edge_of_residue(topology, i1)
    e2 = None if n2 is not None else _find_edge_of_residue(topology, i2)

    start_candidates = _anchor_candidates_for_residue(
        topology,
        sp_graph,
        pos,
        i1,
        bond_length=bond_length,
        segment_length=segment_length,
    )
    end_candidates = _anchor_candidates_for_residue(
        topology,
        sp_graph,
        pos,
        i2,
        bond_length=bond_length,
        segment_length=segment_length,
    )

    best: dict | None = None

    for a1, start_offset, n1, e1, start_loop in start_candidates:
        for a2, end_offset, n2, e2, end_loop in end_candidates:
            try:
                node_path = nx.shortest_path(sp_graph, a1, a2, weight="weight")
                path_cost = nx.shortest_path_length(sp_graph, a1, a2, weight="weight")
            except nx.NetworkXNoPath:
                continue

            total = float(start_offset + path_cost + end_offset)

            if best is None or total < best["total"]:
                best = {
                    "path": node_path,
                    "total": total,
                    "start_offset": float(start_offset),
                    "end_offset": float(end_offset),
                    "n1": n1,
                    "n2": n2,
                    "e1": e1,
                    "e2": e2,
                    "start_loop": start_loop,
                    "end_loop": end_loop,
                }

    if best is None:
        msg = f"No path found between residues {i1} and {i2}"
        raise ValueError(msg)

    node_path: list[AnchorNode] = best["path"]

    edge_path: list[ShortestPathEdgeRef] = [
        _choose_min_weight_edge(sp_graph, u, v)
        for u, v in itertools.pairwise(node_path)
    ]

    fixed_distances: list[float] = []
    random_walk_contour_length = 0.0

    for u, v, k in edge_path:
        data = sp_graph.edges[u, v, k]
        if data["kind"] == "idr":
            # expressed in the same units as the shortest-path graph weight
            random_walk_contour_length += float(data["weight"])
        elif data["kind"] == "domain_shortcut":
            # keep the physical shortcut distance, not the normalized weight
            fixed_distances.append(float(data["distance"]))
        else:
            msg = f"Unknown shortest-path graph edge kind: {data['kind']!r}"
            raise ValueError(msg)

    start_offset = float(best["start_offset"])
    end_offset = float(best["end_offset"])

    # Absorb endpoint offsets into the physical decomposition.
    #
    # - If the endpoint lies in an ordinary IDR edge, the offset is contour length
    #   and should contribute to random_walk_contour_length.
    # - If the endpoint lies in a rigid node, the offset is a rigid shortcut
    #   and should contribute to fixed_distances.
    # - If the endpoint lies in a loop edge, keep the offset separate because
    #   the loop contribution is handled specially via start_loop / end_loop.

    if best["e1"] is not None and best["start_loop"] is None:
        random_walk_contour_length += start_offset
        start_offset = 0.0
    elif (
        best["n1"] is not None
        and best["n1"] not in {"start", "end"}
        and start_offset > 0.0
    ):
        fixed_distances.append(start_offset)
        start_offset = 0.0

    if best["e2"] is not None and best["end_loop"] is None:
        random_walk_contour_length += end_offset
        end_offset = 0.0
    elif (
        best["n2"] is not None
        and best["n2"] not in {"start", "end"}
        and end_offset > 0.0
    ):
        fixed_distances.append(end_offset)
        end_offset = 0.0

    return PathProperties(
        path=node_path,
        edge_path=edge_path,
        total_weight=float(best["total"]),
        start_offset=start_offset,
        end_offset=end_offset,
        fixed_distances=fixed_distances,
        random_walk_contour_length=float(random_walk_contour_length),
        n1=best["n1"],
        n2=best["n2"],
        e1=best["e1"],
        e2=best["e2"],
        start_loop=best["start_loop"],
        end_loop=best["end_loop"],
    )
