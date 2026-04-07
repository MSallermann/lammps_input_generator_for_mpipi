from typing import Any

import networkx as nx
import pytest

from mpipi_lammps_gen.globular_domains import GlobularDomain, protein_topology
from mpipi_lammps_gen.shortest_path_graph import build_shortest_path_graph

# adjust these imports to your package structure
# from your_module import GlobularDomain, protein_topology, build_shortest_path_graph


def assert_has_edge_with_attrs(
    g: nx.MultiGraph,
    u: Any,
    v: Any,
    *,
    kind: str,
    weight: float,
    distance: float,
):
    assert g.has_edge(u, v), f"Expected edge {u!r} -- {v!r}"

    for data in g[u][v].values():
        if (
            data["kind"] == kind
            and data["weight"] == pytest.approx(weight)
            and data["distance"] == pytest.approx(distance)
        ):
            return

    msg = (
        f"No matching edge found between {u!r} and {v!r}. "
        f"Existing edges: {list(g[u][v].values())}"
    )
    raise AssertionError(msg)


def get_matching_edge_data(
    g: nx.MultiGraph,
    u: Any,
    v: Any,
    *,
    kind: str,
):
    assert g.has_edge(u, v), f"Expected edge {u!r} -- {v!r}"

    for data in g[u][v].values():
        if data["kind"] == kind:
            return data

    msg = (
        f"No edge of kind {kind!r} found between {u!r} and {v!r}. "
        f"Existing edges: {list(g[u][v].values())}"
    )

    raise AssertionError(msg)


def test_single_domain_builds_expected_anchor_graph_without_normalization():
    # residues:
    # 0 1 | 2 3 4 | 5 6
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(n_residues=7, domains=[d0])

    positions = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (2.0, 0.0, 0.0),  # 2
        (2.5, 0.0, 0.0),  # 3
        (3.0, 0.0, 0.0),  # 4
        (4.0, 0.0, 0.0),  # 5
        (5.0, 0.0, 0.0),  # 6
    ]

    g = build_shortest_path_graph(
        topology,
        positions,
        bond_length=0.5,
    )

    expected_nodes = {
        ("start", 0),
        ("CD0", 2),
        ("CD0", 4),
        ("end", 6),
    }
    assert set(g.nodes) == expected_nodes

    assert_has_edge_with_attrs(
        g,
        ("start", 0),
        ("CD0", 2),
        kind="idr",
        distance=1.0,
        weight=1.0,
    )
    assert_has_edge_with_attrs(
        g,
        ("CD0", 4),
        ("end", 6),
        kind="idr",
        distance=1.0,
        weight=1.0,
    )
    assert_has_edge_with_attrs(
        g,
        ("CD0", 2),
        ("CD0", 4),
        kind="domain_shortcut",
        distance=1.0,
        weight=1.0,
    )


def test_single_domain_builds_expected_anchor_graph_with_normalization():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(n_residues=7, domains=[d0])

    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
    ]

    g = build_shortest_path_graph(
        topology,
        positions,
        bond_length=0.5,
        segment_length=0.25,
    )

    assert_has_edge_with_attrs(
        g,
        ("start", 0),
        ("CD0", 2),
        kind="idr",
        distance=1.0,
        weight=4.0,
    )
    assert_has_edge_with_attrs(
        g,
        ("CD0", 2),
        ("CD0", 4),
        kind="domain_shortcut",
        distance=1.0,
        weight=4.0,
    )


def test_self_loop_idr_creates_two_anchors_in_same_domain():
    # residues:
    # 0 | 1 2 | 3 4 | 5 6 | 7
    #
    # CD0 = (1,2) and (5,6)
    d0 = GlobularDomain(indices=[(1, 2), (5, 6)])
    topology = protein_topology(n_residues=8, domains=[d0])

    positions = [(float(i), 0.0, 0.0) for i in range(8)]

    g = build_shortest_path_graph(
        topology,
        positions,
        bond_length=0.4,
    )

    expected_nodes = {
        ("start", 0),
        ("CD0", 1),
        ("CD0", 2),
        ("CD0", 5),
        ("CD0", 6),
        ("end", 7),
    }
    assert set(g.nodes) == expected_nodes

    # self-loop IDR over residues 3..4 has length 2 => distance 0.8
    assert_has_edge_with_attrs(
        g,
        ("CD0", 2),
        ("CD0", 5),
        kind="idr",
        distance=0.8,
        weight=0.8,
    )

    # the same anchor pair may also have a domain shortcut edge
    assert_has_edge_with_attrs(
        g,
        ("CD0", 2),
        ("CD0", 5),
        kind="domain_shortcut",
        distance=3.0,
        weight=3.0,
    )

    assert_has_edge_with_attrs(
        g,
        ("CD0", 1),
        ("CD0", 2),
        kind="domain_shortcut",
        distance=1.0,
        weight=1.0,
    )
    assert_has_edge_with_attrs(
        g,
        ("CD0", 5),
        ("CD0", 6),
        kind="domain_shortcut",
        distance=1.0,
        weight=1.0,
    )


def test_parallel_idrs_become_distinct_edges_between_distinct_anchor_nodes():
    d0 = GlobularDomain(indices=[(0, 0), (4, 4)])
    d1 = GlobularDomain(indices=[(2, 2), (6, 6)])
    topology = protein_topology(n_residues=7, domains=[d0, d1])

    positions = [(float(i), 0.0, 0.0) for i in range(7)]

    g = build_shortest_path_graph(
        topology,
        positions,
        bond_length=0.3,
    )

    assert_has_edge_with_attrs(
        g,
        ("CD0", 0),
        ("CD1", 2),
        kind="idr",
        distance=0.3,
        weight=0.3,
    )
    assert_has_edge_with_attrs(
        g,
        ("CD1", 2),
        ("CD0", 4),
        kind="idr",
        distance=0.3,
        weight=0.3,
    )
    assert_has_edge_with_attrs(
        g,
        ("CD0", 4),
        ("CD1", 6),
        kind="idr",
        distance=0.3,
        weight=0.3,
    )


def test_domain_shortcuts_are_complete_graph_over_anchors():
    d0 = GlobularDomain(indices=[(1, 2), (5, 6)])
    topology = protein_topology(n_residues=8, domains=[d0])
    positions = [(float(i), 0.0, 0.0) for i in range(8)]

    g = build_shortest_path_graph(
        topology,
        positions,
        bond_length=0.4,
    )

    cd0_anchors = [("CD0", 1), ("CD0", 2), ("CD0", 5), ("CD0", 6)]

    for i, a1 in enumerate(cd0_anchors):
        for a2 in cd0_anchors[i + 1 :]:
            assert g.has_edge(a1, a2), f"Missing shortcut edge {a1} -- {a2}"


def test_shortest_path_uses_both_idr_and_domain_shortcut_costs():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(n_residues=7, domains=[d0])

    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
    ]

    g = build_shortest_path_graph(
        topology,
        positions,
        bond_length=0.5,
    )

    source = ("start", 0)
    target = ("end", 6)

    path = nx.shortest_path(g, source=source, target=target, weight="weight")
    cost = nx.shortest_path_length(g, source=source, target=target, weight="weight")

    assert path == [
        ("start", 0),
        ("CD0", 2),
        ("CD0", 4),
        ("end", 6),
    ]
    assert cost == pytest.approx(3.0)


def test_idr_edge_attributes_are_preserved():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(n_residues=7, domains=[d0])
    positions = [(float(i), 0.0, 0.0) for i in range(7)]

    g = build_shortest_path_graph(
        topology,
        positions,
        bond_length=0.5,
    )

    data = get_matching_edge_data(g, ("start", 0), ("CD0", 2), kind="idr")

    assert data["kind"] == "idr"
    assert data["start_idx"] == 0
    assert data["end_idx"] == 1
    assert data["length"] == 2
    assert data["loop"] is False
    assert "topology_edge" in data


def test_domain_shortcut_attributes_are_preserved():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(n_residues=7, domains=[d0])
    positions = [(float(i), 0.0, 0.0) for i in range(7)]

    g = build_shortest_path_graph(
        topology,
        positions,
        bond_length=0.5,
    )

    data = get_matching_edge_data(g, ("CD0", 2), ("CD0", 4), kind="domain_shortcut")

    assert data["kind"] == "domain_shortcut"
    assert data["topology_node"] == "CD0"
    assert data["idx1"] == 2
    assert data["idx2"] == 4


def test_invalid_residue_positions_shape_raises():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(n_residues=7, domains=[d0])

    with pytest.raises(ValueError, match="shape"):
        build_shortest_path_graph(
            topology,
            residue_positions=[(0.0, 0.0)],
            bond_length=0.5,
        )


def test_nonpositive_bond_length_raises():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(n_residues=7, domains=[d0])
    positions = [(float(i), 0.0, 0.0) for i in range(7)]

    with pytest.raises(ValueError, match="bond_length"):
        build_shortest_path_graph(
            topology,
            positions,
            bond_length=0.0,
        )


def test_nonpositive_segment_length_raises():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(n_residues=7, domains=[d0])
    positions = [(float(i), 0.0, 0.0) for i in range(7)]

    with pytest.raises(ValueError, match="segment_length"):
        build_shortest_path_graph(
            topology,
            positions,
            bond_length=0.5,
            segment_length=0.0,
        )
