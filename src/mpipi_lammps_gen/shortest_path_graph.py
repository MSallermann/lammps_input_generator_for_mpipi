from __future__ import annotations

from collections import defaultdict
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

    return path_graph
