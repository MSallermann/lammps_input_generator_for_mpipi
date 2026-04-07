import networkx as nx
import pytest

from mpipi_lammps_gen.globular_domains import (
    GlobularDomain,
    protein_topology,
)


def edge_tuples_with_data(g: nx.MultiGraph):
    """
    Return edges in a comparison-friendly form.
    """
    out = []
    for u, v, _, d in g.edges(keys=True, data=True):
        out.append(
            (
                u,
                v,
                {
                    "start_idx": d["start_idx"],
                    "end_idx": d["end_idx"],
                    "length": d["length"],
                    "weight": d["weight"],
                    "loop": d["loop"],
                },
            )
        )
    return out


def assert_has_edge(g: nx.MultiGraph, u: str, v: str, **attrs):
    """
    Assert that at least one multiedge between u and v has the given attrs.
    """
    if not g.has_edge(u, v):
        msg = f"No edge between {u!r} and {v!r}"
        raise AssertionError(msg)

    for data in g[u][v].values():
        if all(data.get(k) == val for k, val in attrs.items()):
            return

    msg = (
        f"No edge between {u!r} and {v!r} matched attrs {attrs!r}. "
        f"Existing edges: {list(g[u][v].values())}"
    )

    raise AssertionError(msg)


def test_idp_only_start_to_end():
    g = protein_topology(n_residues=5, domains=[])

    assert isinstance(g, nx.MultiGraph)

    assert set(g.nodes) == {"start", "end"}
    assert g.nodes["start"]["indices"] == {0}
    assert g.nodes["end"]["indices"] == {4}

    assert g.number_of_edges() == 1
    assert_has_edge(
        g,
        "start",
        "end",
        start_idx=0,
        end_idx=4,
        length=5,
        weight=1.0 / 5,
        loop=False,
    )


def test_single_domain_with_idrs_on_both_sides():
    # residues: 0 1 | 2 3 4 | 5 6
    # start-IDR 0..1, domain 2..4, end-IDR 5..6
    d0 = GlobularDomain(indices=[(2, 4)])

    g = protein_topology(n_residues=7, domains=[d0])

    assert set(g.nodes) == {"start", "end", "CD0"}
    assert g.nodes["CD0"]["indices"] == {2, 3, 4}

    assert g.number_of_edges() == 2
    assert_has_edge(
        g,
        "start",
        "CD0",
        start_idx=0,
        end_idx=1,
        length=2,
        weight=1.0 / 2,
        loop=False,
    )
    assert_has_edge(
        g,
        "CD0",
        "end",
        start_idx=5,
        end_idx=6,
        length=2,
        weight=1.0 / 2,
        loop=False,
    )


def test_domain_starts_at_zero_no_start_edge():
    # residues: | 0 1 2 | 3 4
    d0 = GlobularDomain(indices=[(0, 2)])

    g = protein_topology(n_residues=5, domains=[d0])

    assert g.number_of_edges() == 1
    assert not g.has_edge("start", "CD0")
    assert_has_edge(
        g,
        "CD0",
        "end",
        start_idx=3,
        end_idx=4,
        length=2,
        weight=1.0 / 2,
        loop=False,
    )


def test_domain_ends_at_last_residue_no_end_edge():
    # residues: 0 1 | 2 3 4
    d0 = GlobularDomain(indices=[(2, 4)])

    g = protein_topology(n_residues=5, domains=[d0])

    assert g.number_of_edges() == 1
    assert_has_edge(
        g,
        "start",
        "CD0",
        start_idx=0,
        end_idx=1,
        length=2,
        weight=1.0 / 2,
        loop=False,
    )
    assert not g.has_edge("CD0", "end")


def test_adjacent_domains_create_no_edge_between_them():
    # residues: 0 | 1 2 | 3 4 | 5
    d0 = GlobularDomain(indices=[(1, 2)])
    d1 = GlobularDomain(indices=[(3, 4)])

    g = protein_topology(n_residues=6, domains=[d0, d1])

    # start -> CD0 for residue 0
    assert_has_edge(
        g,
        "start",
        "CD0",
        start_idx=0,
        end_idx=0,
        length=1,
        weight=1.0,
        loop=False,
    )

    # no IDR between adjacent domains
    assert not g.has_edge("CD0", "CD1")

    # CD1 -> end for residue 5
    assert_has_edge(
        g,
        "CD1",
        "end",
        start_idx=5,
        end_idx=5,
        length=1,
        weight=1.0,
        loop=False,
    )


def test_multiple_edges_between_same_pair_of_nodes():
    # one domain split into two rigid pieces gives a self-loop IDR;
    # plus start and end IDRs
    #
    # residues:
    # 0 | 1 2 | 3 4 | 5 6 | 7
    #
    # domain CD0 = (1,2) and (5,6)
    # IDRs:
    #   start -> CD0 : 0..0
    #   CD0 -> CD0   : 3..4   (self-loop)
    #   CD0 -> end   : 7..7
    d0 = GlobularDomain(indices=[(1, 2), (5, 6)])

    g = protein_topology(n_residues=8, domains=[d0])

    assert g.number_of_edges() == 3

    assert_has_edge(
        g,
        "start",
        "CD0",
        start_idx=0,
        end_idx=0,
        length=1,
        weight=1.0,
        loop=False,
    )
    assert_has_edge(
        g,
        "CD0",
        "CD0",
        start_idx=3,
        end_idx=4,
        length=2,
        weight=1.0 / 2,
        loop=True,
    )
    assert_has_edge(
        g,
        "CD0",
        "end",
        start_idx=7,
        end_idx=7,
        length=1,
        weight=1.0,
        loop=False,
    )


def test_two_distinct_idrs_between_same_two_domains_create_multiedges():
    # residues:
    #   0 1 | 2 | 3 | 4 | 5 6
    #
    # CD0 = residues 0,1 and 4
    # CD1 = residues 2 and 5,6
    #
    # IDRs:
    #   CD1 <-> CD0 over residue 3
    # and depending on order/structure we want repeated connections possible.
    #
    # A cleaner explicit example:
    # CD0 has two rigid blocks, CD1 has two rigid blocks, creating two separate
    # disordered stretches connecting the same pair of nodes.
    d0 = GlobularDomain(indices=[(0, 0), (4, 4)])
    d1 = GlobularDomain(indices=[(2, 2), (6, 6)])

    g = protein_topology(n_residues=7, domains=[d0, d1])

    # Traversal:
    # 0 CD0
    # 1 IDR   => CD0--CD1
    # 2 CD1
    # 3 IDR   => CD1--CD0
    # 4 CD0
    # 5 IDR   => CD0--CD1
    # 6 CD1
    #
    # Because this is a MultiGraph, repeated connections are allowed.
    # There should be:
    #   one CD0-CD1 edge for residue 1
    #   one CD1-CD0 edge for residue 3  (same undirected node pair)
    #   one CD0-CD1 edge for residue 5
    #
    # In an undirected MultiGraph, all three are between CD0 and CD1.
    assert g.number_of_edges("CD0", "CD1") == 3

    assert_has_edge(
        g,
        "CD0",
        "CD1",
        start_idx=1,
        end_idx=1,
        length=1,
        weight=1.0,
        loop=False,
    )
    assert_has_edge(
        g,
        "CD0",
        "CD1",
        start_idx=3,
        end_idx=3,
        length=1,
        weight=1.0,
        loop=False,
    )
    assert_has_edge(
        g,
        "CD0",
        "CD1",
        start_idx=5,
        end_idx=5,
        length=1,
        weight=1.0,
        loop=False,
    )


def test_overlapping_domains_raise():
    d0 = GlobularDomain(indices=[(1, 3)])
    d1 = GlobularDomain(indices=[(3, 5)])

    with pytest.raises(ValueError, match="belongs to multiple domains"):
        protein_topology(n_residues=7, domains=[d0, d1])


def test_node_attributes_are_correct():
    d0 = GlobularDomain(indices=[(1, 2), (5, 5)])

    g = protein_topology(n_residues=7, domains=[d0])

    assert g.nodes["start"]["weight"] == 1
    assert g.nodes["start"]["indices"] == {0}

    assert g.nodes["end"]["weight"] == 1
    assert g.nodes["end"]["indices"] == {6}

    assert g.nodes["CD0"]["weight"] == d0.n_atoms()
    assert g.nodes["CD0"]["indices"] == {1, 2, 5}
