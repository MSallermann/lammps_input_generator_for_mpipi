import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import networkx as nx
import numpy as np

from mpipi_lammps_gen.shortest_path_graph import (
    PathProperties,
    _anchor_candidates_for_residue,
    _choose_min_weight_edge,
    _find_contour_edge_of_residue,
    _find_edge_of_residue,
    _find_node_of_residue,
    build_shortest_path_graph,
)

AnchorNode: TypeAlias = tuple[str, int]
TopologyEdgeRef: TypeAlias = tuple[str, str, int]
ShortestPathEdgeRef: TypeAlias = tuple[AnchorNode, AnchorNode, int]


@dataclass(frozen=True)
class ResidueInfo:
    node: str | None
    edge: TopologyEdgeRef | None
    contour_edge: TopologyEdgeRef | None
    candidates: tuple[
        tuple[
            AnchorNode,
            float,  # offset cost
            str | None,  # containing node
            TopologyEdgeRef | None,  # containing edge
            tuple[int, int] | None,  # loop metadata
        ],
        ...,
    ]


@dataclass
class PathQueryCache:
    topology: nx.MultiGraph
    shortest_path_graph: nx.MultiGraph
    residue_positions: np.ndarray
    bond_length: float
    segment_length: float | None
    residue_info: list[ResidueInfo]
    anchor_path_lengths: dict[AnchorNode, dict[AnchorNode, float]]
    anchor_paths: dict[AnchorNode, dict[AnchorNode, list[AnchorNode]]]
    min_edge_by_pair: dict[tuple[AnchorNode, AnchorNode], ShortestPathEdgeRef]


def build_path_query_cache(
    topology: nx.MultiGraph,
    residue_positions: Sequence[Sequence[float]],
    *,
    shortest_path_graph: nx.MultiGraph | None = None,
    bond_length: float,
    segment_length: float | None = None,
) -> PathQueryCache:
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

    residue_info: list[ResidueInfo] = []

    for i in range(len(pos)):
        n = _find_node_of_residue(topology, i)
        e = _find_edge_of_residue(topology, i)
        contour_e = _find_contour_edge_of_residue(topology, i)

        candidates = _anchor_candidates_for_residue(
            topology,
            sp_graph,
            pos,
            i,
            bond_length=bond_length,
            segment_length=segment_length,
        )

        residue_info.append(
            ResidueInfo(
                node=n,
                edge=e,
                contour_edge=contour_e,
                candidates=tuple(candidates),
            )
        )

    anchor_path_lengths: dict[AnchorNode, dict[AnchorNode, float]] = {}
    anchor_paths: dict[AnchorNode, dict[AnchorNode, list[AnchorNode]]] = {}

    for source in sp_graph.nodes:
        anchor_path_lengths[source] = nx.single_source_dijkstra_path_length(
            sp_graph,
            source,
            weight="weight",
        )
        anchor_paths[source] = nx.single_source_dijkstra_path(
            sp_graph,
            source,
            weight="weight",
        )

    min_edge_by_pair: dict[tuple[AnchorNode, AnchorNode], ShortestPathEdgeRef] = {}
    for u, v in sp_graph.edges():
        edge_ref = _choose_min_weight_edge(sp_graph, u, v)
        min_edge_by_pair[(u, v)] = edge_ref
        min_edge_by_pair[(v, u)] = (edge_ref[1], edge_ref[0], edge_ref[2])

    return PathQueryCache(
        topology=topology,
        shortest_path_graph=sp_graph,
        residue_positions=pos,
        bond_length=bond_length,
        segment_length=segment_length,
        residue_info=residue_info,
        anchor_path_lengths=anchor_path_lengths,
        anchor_paths=anchor_paths,
        min_edge_by_pair=min_edge_by_pair,
    )


def get_path_properties_cached(  # noqa: PLR0912, PLR0915
    cache: PathQueryCache,
    i1: int,
    i2: int,
) -> PathProperties:
    pos = cache.residue_positions
    topology = cache.topology
    sp_graph = cache.shortest_path_graph
    segment_length = cache.segment_length
    bond_length = cache.bond_length

    if i1 < 0 or i2 < 0:
        msg = "Residue indices must be non-negative"
        raise ValueError(msg)

    if i1 >= len(pos) or i2 >= len(pos):
        msg = (
            f"Residue indices must be smaller than the number of residues ({len(pos)})"
        )
        raise ValueError(msg)

    r1 = cache.residue_info[i1]
    r2 = cache.residue_info[i2]

    # Case 1: same rigid domain
    if r1.node is not None and r1.node == r2.node and r1.node not in {"start", "end"}:
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
            n1=r1.node,
            n2=r2.node,
            e1=None,
            e2=None,
            start_loop=None,
            end_loop=None,
        )

    # Case 2: same contour edge
    if r1.contour_edge is not None and r1.contour_edge == r2.contour_edge:
        u, v, k = r1.contour_edge
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
            n1=r1.node,
            n2=r2.node,
            e1=r1.contour_edge,
            e2=r2.contour_edge,
            start_loop=start_loop,
            end_loop=end_loop,
        )

    best: dict | None = None

    for a1, start_offset, n1, e1, start_loop in r1.candidates:
        for a2, end_offset, n2, e2, end_loop in r2.candidates:
            if a2 not in cache.anchor_path_lengths[a1]:
                continue

            path_cost = cache.anchor_path_lengths[a1][a2]
            node_path = cache.anchor_paths[a1][a2]
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
        cache.min_edge_by_pair[(u, v)] for u, v in itertools.pairwise(node_path)
    ]

    fixed_distances: list[float] = []
    random_walk_contour_length = 0.0

    for u, v, k in edge_path:
        data = sp_graph.edges[u, v, k]
        if data["kind"] == "idr":
            random_walk_contour_length += float(data["weight"])
        elif data["kind"] == "domain_shortcut":
            fixed_distances.append(float(data["distance"]))
        else:
            msg = f"Unknown shortest-path graph edge kind: {data['kind']!r}"
            raise ValueError(msg)

    start_offset = float(best["start_offset"])
    end_offset = float(best["end_offset"])

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
