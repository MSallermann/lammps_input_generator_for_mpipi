import numpy as np
import pytest

from mpipi_lammps_gen.globular_domains import GlobularDomain, protein_topology
from mpipi_lammps_gen.shortest_path_graph import get_path_properties


def test_path_between_two_terminal_idr_residues():
    # residues:
    #
    # 0 ─► 1 ─► 2 ─► 3 ─► 4
    #
    # there are no rigid domains, so the whole protein is one IDR.
    #
    # path from 1 to 3 is entirely inside that same IDR:
    #
    #   1 --(bond_length)--> 2 --(bond_length)--> 3
    #
    # so:
    #   random_walk_contour_length = 2 * bond_length
    #   fixed_distances            = []
    #   start_offset               = 0
    #   end_offset                 = 0
    #   total_weight               = 2 * bond_length

    topology = protein_topology(
        n_residues=5,
        domains=[],
    )

    bond_length = 0.5

    positions = [(float(i) * bond_length, 0.0, 0.0) for i in range(5)]

    props = get_path_properties(
        topology,
        i1=1,
        i2=3,
        residue_positions=positions,
        bond_length=bond_length,
    )

    # since both residues lie in the same IDR, this is handled directly
    # without going through the anchor graph
    assert props.path == []
    assert props.edge_path == []

    # no endpoint attachment bookkeeping is needed in this direct case
    assert props.start_offset == pytest.approx(0.0)
    assert props.end_offset == pytest.approx(0.0)

    # there are no rigid shortcuts
    assert props.fixed_distances == []

    # contour distance from 1 to 3 is two bonds
    assert props.random_walk_contour_length == pytest.approx(2.0 * bond_length)

    # total weight is just that contour distance
    assert props.total_weight == pytest.approx(2.0 * bond_length)


def test_path_from_rigid_domain_to_rigid_domain_uses_domain_shortcut_and_idrs():
    # residues:
    #
    # 0 ─► 1 ─► [2 ─► 3 ─► 4] ─► 5 ─► 6
    #            rigid domain CD0
    #
    # terminal IDRs:
    #   left  IDR = 0, 1
    #   right IDR = 5, 6
    #
    # path from 0 to 6 goes:
    #
    #   0 --(IDR)--> 2 --(rigid shortcut)--> 4 --(IDR)--> 6
    #
    # more explicitly:
    #   0 -> 1 -> 2       contributes 2 * bond_length
    #   2 -> 4            contributes Euclidean shortcut distance
    #   4 -> 5 -> 6       contributes 2 * bond_length
    #
    # so:
    #   random_walk_contour_length = 4 * bond_length
    #   fixed_distances            = [distance(2,4)] = [2 * bond_length]
    #   start_offset               = 0   (0 is already the start anchor)
    #   end_offset                 = 0   (6 is already the end anchor)
    #   total_weight               = 6 * bond_length

    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(
        n_residues=7,
        domains=[d0],
    )

    bond_length = 0.5

    positions = [
        (0.0 * bond_length, 0.0, 0.0),  # 0
        (1.0 * bond_length, 0.0, 0.0),  # 1
        (2.0 * bond_length, 0.0, 0.0),  # 2
        (3.0 * bond_length, 0.0, 0.0),  # 3
        (4.0 * bond_length, 0.0, 0.0),  # 4
        (5.0 * bond_length, 0.0, 0.0),  # 5
        (6.0 * bond_length, 0.0, 0.0),  # 6
    ]

    props = get_path_properties(
        topology,
        i1=0,
        i2=6,
        residue_positions=positions,
        bond_length=bond_length,
    )

    # the anchor-graph path goes from the start anchor, through the two domain anchors,
    # to the end anchor
    assert props.path == [
        ("start", 0),
        ("CD0", 2),
        ("CD0", 4),
        ("end", 6),
    ]

    # this path consists of:
    #   start -> CD0(2)   : IDR
    #   CD0(2) -> CD0(4)  : rigid shortcut
    #   CD0(4) -> end     : IDR
    assert len(props.edge_path) == 3

    # start and end residues are already anchors
    assert props.start_offset == pytest.approx(0.0)
    assert props.end_offset == pytest.approx(0.0)

    # IDRs: 0 -> 2 and 4 -> 6, each contributes 2 * bond_length
    assert props.random_walk_contour_length == pytest.approx(4.0 * bond_length)

    # rigid shortcut inside the domain: residue 2 -> residue 4
    assert props.fixed_distances == pytest.approx([2.0 * bond_length])

    # total weight = 2*bond_length + 2*bond_length + 2*bond_length
    assert props.total_weight == pytest.approx(6.0 * bond_length)

    # both endpoints lie in topology nodes, not topology edges
    assert props.n1 == "start"
    assert props.n2 == "end"
    assert props.e1 is None
    assert props.e2 is None

    # neither endpoint lies in a loop
    assert props.start_loop is None
    assert props.end_loop is None


def test_path_between_two_residues_inside_same_domain():
    # residues:
    #
    # 0 ─► 1 ─► [2 ─► 3 ─► 4] ─► 5 ─► 6
    #            rigid domain CD0
    #
    # both residues 2 and 4 lie inside the same rigid domain.
    #
    # so the path is handled directly inside that domain, without using the
    # anchor graph:
    #
    #   2 --(rigid shortcut)--> 4
    #
    # therefore:
    #   random_walk_contour_length = 0
    #   fixed_distances            = [distance(2,4)]
    #   start_offset               = 0
    #   end_offset                 = 0
    #   total_weight               = distance(2,4)

    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(
        n_residues=7,
        domains=[d0],
    )

    bond_length = 0.5

    positions = [(float(i) * bond_length, 0.0, 0.0) for i in range(7)]

    props = get_path_properties(
        topology,
        i1=2,
        i2=4,
        residue_positions=positions,
        bond_length=bond_length,
    )

    # same rigid domain: direct shortcut, no anchor-graph traversal needed
    assert props.path == []
    assert props.edge_path == []

    # no endpoint attachment bookkeeping is needed in this direct case
    assert props.start_offset == pytest.approx(0.0)
    assert props.end_offset == pytest.approx(0.0)

    # there is no IDR contribution
    assert props.random_walk_contour_length == pytest.approx(0.0)

    # rigid shortcut from residue 2 to residue 4
    assert props.fixed_distances == pytest.approx([2.0 * bond_length])

    # total weight is just that rigid shortcut distance
    assert props.total_weight == pytest.approx(2.0 * bond_length)

    # both endpoints lie in the rigid node CD0, not in topology edges
    assert props.n1 == "CD0"
    assert props.n2 == "CD0"
    assert props.e1 is None
    assert props.e2 is None

    # neither endpoint lies in a loop
    assert props.start_loop is None
    assert props.end_loop is None


def test_start_residue_inside_self_loop_records_start_loop_metadata():
    # residues:
    #     ┌──────┐
    #  0─►│1 ─► 2├─►3
    #     │      │  ▼
    #  7◄─┤6 ◄─ 5│◄─4
    #     └──────┘

    d0 = GlobularDomain(indices=[(1, 2), (5, 6)])
    topology = protein_topology(
        n_residues=8,
        domains=[d0],
    )

    bond_length = 0.5

    positions = [
        (-1.0 * bond_length, 1.0 * bond_length, 0.0),  # 0
        (0.0 * bond_length, 1.0 * bond_length, 0.0),  # 1
        (1.0 * bond_length, 1.0 * bond_length, 0.0),  # 2
        (2.0 * bond_length, 1.0 * bond_length, 0.0),  # 3
        (2.0 * bond_length, 0.0 * bond_length, 0.0),  # 4
        (1.0 * bond_length, 0.0 * bond_length, 0.0),  # 5
        (0.0 * bond_length, 0.0 * bond_length, 0.0),  # 6
        (-1.0 * bond_length, 0.0 * bond_length, 0.0),  # 7
    ]

    props = get_path_properties(
        topology,
        i1=3,  # inside self-loop IDR 3..4
        i2=7,
        residue_positions=positions,
        bond_length=bond_length,
    )

    # We do start in an edge
    assert props.e1 is not None
    # Which is indeed a self loop
    assert topology.edges[props.e1]["loop"] is True

    # we are on the first residue of the loop
    # and the loop consists of two residues (and three segments)
    assert props.start_loop == (0, 2)

    # we do not end in a loop
    assert props.end_loop is None

    # the only random segment is 6->7
    assert props.random_walk_contour_length == 1.0 * bond_length

    # residue 3 is one bond away from the left loop anchor at residue 2
    assert props.start_offset == pytest.approx(1.0 * bond_length)

    # we end at seven, which is the terminus of the protein
    assert props.path[-1] == ("end", 7)

    assert props.total_weight > 0.0


def test_end_residue_inside_self_loop_records_end_loop_metadata():
    # residues:
    #     ┌──────┐
    #  0─►│1 ─► 2├─►3
    #     │      │  ▼
    #  7◄─┤6 ◄─ 5│◄─4
    #     └──────┘

    d0 = GlobularDomain(indices=[(1, 2), (5, 6)])
    topology = protein_topology(
        n_residues=8,
        domains=[d0],
    )

    bond_length = 0.5

    positions = [
        (-1.0 * bond_length, 1.0 * bond_length, 0.0),  # 0
        (0.0 * bond_length, 1.0 * bond_length, 0.0),  # 1
        (1.0 * bond_length, 1.0 * bond_length, 0.0),  # 2
        (2.0 * bond_length, 1.0 * bond_length, 0.0),  # 3
        (2.0 * bond_length, 0.0 * bond_length, 0.0),  # 4
        (1.0 * bond_length, 0.0 * bond_length, 0.0),  # 5
        (0.0 * bond_length, 0.0 * bond_length, 0.0),  # 6
        (-1.0 * bond_length, 0.0 * bond_length, 0.0),  # 7
    ]

    props = get_path_properties(
        topology,
        i1=0,
        i2=4,  # inside self-loop IDR 3..4
        residue_positions=positions,
        bond_length=bond_length,
    )

    # We start at residue 0, which is not inside a loop
    assert props.start_loop is None

    # We end in an edge
    assert props.e2 is not None
    # Which is indeed a self loop
    assert topology.edges[props.e2]["loop"] is True

    # we are on the second residue of the loop (index 1 of 2)
    # and the loop consists of two residues (and three segments)
    assert props.end_loop == (1, 2)

    # the only random segment is 0 -> 1
    assert props.random_walk_contour_length == pytest.approx(1.0 * bond_length)

    # residue 4 is one bond away from the right loop anchor at residue 5
    assert props.end_offset == pytest.approx(1.0 * bond_length)

    # start residue is already at the start anchor
    assert props.start_offset == pytest.approx(0.0)

    # rigid shortcut from residue 1 to residue 5
    # distance = sqrt(2) * bond_length
    assert props.fixed_distances == pytest.approx([np.sqrt(2.0) * bond_length])

    # we start at the start anchor
    assert props.path[0] == ("start", 0)

    # total weight = bond_length + sqrt(2)*bond_length + bond_length
    assert props.total_weight == pytest.approx((2.0 + np.sqrt(2.0)) * bond_length)


def test_segment_length_normalizes_total_weight_and_idr_contribution():
    # residues:
    #
    # 0 ─► 1 ─► [2 ─► 3 ─► 4] ─► 5 ─► 6
    #            rigid domain CD0
    #
    # path from 0 to 6:
    #
    #   0 -> 1 -> 2        (IDR: 2 bonds)
    #   2 -> 4             (rigid shortcut)
    #   4 -> 5 -> 6        (IDR: 2 bonds)
    #
    # without normalization:
    #   random_walk_contour_length = 4 * bond_length = 2.0
    #   fixed_distances            = [distance(2,4)] = [2 * bond_length] = [1.0]
    #   total_weight               = 3.0
    #
    # with segment_length = 0.25:
    #   all weights are divided by 0.25
    #
    #   total_weight               = 3.0 / 0.25 = 12.0
    #   random_walk_contour_length = 2.0 / 0.25 = 8.0
    #
    # fixed_distances remain physical distances (not normalized)

    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(
        n_residues=7,
        domains=[d0],
    )

    bond_length = 0.5

    positions = [(float(i) * bond_length, 0.0, 0.0) for i in range(7)]

    props = get_path_properties(
        topology,
        i1=0,
        i2=6,
        residue_positions=positions,
        bond_length=bond_length,
        segment_length=0.25,
    )

    # total weight is normalized by segment_length
    assert props.total_weight == pytest.approx(12.0)

    # only IDR contributions are accumulated here, also normalized
    assert props.random_walk_contour_length == pytest.approx(8.0)

    # rigid shortcut distance remains a physical distance
    assert props.fixed_distances == pytest.approx([2.0 * bond_length])


def test_endpoint_idr_offset_is_absorbed_into_random_walk_contour_length():
    # residues:
    #
    # 0 ─► 1 ─► [2 ─► 3 ─► 4] ─► 5 ─► 6
    #            rigid domain CD0
    #
    # path from 0 to 5 should be:
    #
    #   0 -> 1 -> 2        (IDR: 2 bonds)
    #   2 -> 4             (rigid shortcut)
    #   4 -> 5             (IDR: 1 bond)
    #
    # so physically:
    #   random_walk_contour_length = 3 * bond_length
    #   fixed_distances            = [distance(2,4)] = [2 * bond_length]
    #   start_offset               = 0
    #   end_offset                 = 0
    #   total_weight               = 5 * bond_length
    #
    # This case forces the anchor-graph branch:
    #   - residue 0 is the start anchor
    #   - residue 5 lies inside the right-hand IDR
    #
    # If endpoint offsets are not absorbed into the physical decomposition,
    # the implementation will typically return:
    #   random_walk_contour_length = 2 * bond_length
    #   end_offset                 = 1 * bond_length
    #
    # which is the bug we want to catch.

    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(
        n_residues=7,
        domains=[d0],
    )

    bond_length = 0.5

    positions = [(float(i) * bond_length, 0.0, 0.0) for i in range(7)]

    props = get_path_properties(
        topology,
        i1=0,
        i2=5,  # inside the right-hand IDR adjacent to CD0
        residue_positions=positions,
        bond_length=bond_length,
    )

    # the path should go from the start anchor, through the left domain anchor,
    # through the right domain anchor, and then terminate at residue 5 via the
    # same contour segment that connects 4 -> 5 -> 6
    assert props.path == [
        ("start", 0),
        ("CD0", 2),
        ("CD0", 4),
    ]

    # physically, the contour length should include both IDR pieces:
    #   0 -> 1 -> 2   and   4 -> 5
    assert props.random_walk_contour_length == pytest.approx(3.0 * bond_length)

    # the rigid shortcut is 2 -> 4
    assert props.fixed_distances == pytest.approx([2.0 * bond_length])

    # once the endpoint contour segment has been absorbed into the decomposition,
    # there should be no leftover endpoint bookkeeping
    assert props.start_offset == pytest.approx(0.0)
    assert props.end_offset == pytest.approx(0.0)

    # total = 3*b + 2*b = 5*b
    assert props.total_weight == pytest.approx(5.0 * bond_length)


def test_last_residue_in_rigid_domain_does_not_get_misclassified_as_end():
    # residues:
    #
    # 0 ─► 1 ─► 2 ─► [3 ─► 4 ─► 5]
    #                  rigid domain CD0
    #
    # residue 5 is both:
    #   - the terminal residue of the protein
    #   - part of the rigid domain CD0
    #
    # physically, it should be treated as belonging to CD0, not as a bare "end" node.
    #
    # with the current bug, get_path_properties(...) may classify residue 5 as "end",
    # then fail because no "end" anchor exists in the shortest-path graph.

    d0 = GlobularDomain(indices=[(3, 5)])
    topology = protein_topology(
        n_residues=6,
        domains=[d0],
    )

    bond_length = 0.5
    positions = [(float(i) * bond_length, 0.0, 0.0) for i in range(6)]

    props = get_path_properties(
        topology,
        i1=0,
        i2=5,
        residue_positions=positions,
        bond_length=bond_length,
    )

    # the last residue lies inside the rigid domain CD0
    assert props.n2 == "CD0"
    assert props.e2 is None

    # path from 0 to 5 is:
    #   0 -> 1 -> 2 -> 3   (IDR: 3 bonds)
    #   3 -> 5             (rigid shortcut)
    #
    # so:
    #   random_walk_contour_length = 3 * bond_length
    #   fixed_distances            = [distance(3,5)] = [2 * bond_length]
    #   end_offset                 = 0
    assert props.random_walk_contour_length == pytest.approx(3.0 * bond_length)
    assert props.fixed_distances == pytest.approx([2.0 * bond_length])
    assert props.end_offset == pytest.approx(0.0)


def test_first_residue_in_rigid_domain_does_not_get_misclassified_as_start():
    # residues:
    #
    # [0 ─► 1 ─► 2] ─► 3 ─► 4 ─► 5
    #   rigid domain CD0
    #
    # residue 0 is both:
    #   - the first residue of the protein
    #   - part of the rigid domain CD0
    #
    # physically, it should be treated as belonging to CD0, not as a bare "start" node.

    d0 = GlobularDomain(indices=[(0, 2)])
    topology = protein_topology(
        n_residues=6,
        domains=[d0],
    )

    bond_length = 0.5
    positions = [(float(i) * bond_length, 0.0, 0.0) for i in range(6)]

    props = get_path_properties(
        topology,
        i1=0,
        i2=5,
        residue_positions=positions,
        bond_length=bond_length,
    )

    # the first residue lies inside the rigid domain CD0
    assert props.n1 == "CD0"
    assert props.e1 is None

    # path from 0 to 5 is:
    #   0 -> 2             (rigid shortcut)
    #   2 -> 3 -> 4 -> 5   (IDR: 3 bonds)
    #
    # so:
    #   random_walk_contour_length = 3 * bond_length
    #   fixed_distances            = [distance(0,2)] = [2 * bond_length]
    #   start_offset               = 0
    assert props.random_walk_contour_length == pytest.approx(3.0 * bond_length)
    assert props.fixed_distances == pytest.approx([2.0 * bond_length])
    assert props.start_offset == pytest.approx(0.0)


def test_negative_residue_index_raises():
    topology = protein_topology(
        n_residues=5,
        domains=[],
    )
    positions = [(float(i), 0.0, 0.0) for i in range(5)]

    with pytest.raises(ValueError, match="non-negative"):
        get_path_properties(
            topology,
            i1=-1,
            i2=3,
            residue_positions=positions,
            bond_length=0.5,
        )


def test_out_of_range_residue_index_raises():
    topology = protein_topology(
        n_residues=5,
        domains=[],
    )
    positions = [(float(i), 0.0, 0.0) for i in range(5)]

    with pytest.raises(ValueError, match="smaller than the number of residues"):
        get_path_properties(
            topology,
            i1=0,
            i2=5,
            residue_positions=positions,
            bond_length=0.5,
        )


def test_invalid_positions_shape_raises():
    topology = protein_topology(
        n_residues=5,
        domains=[],
    )

    with pytest.raises(ValueError, match="shape"):
        get_path_properties(
            topology,
            i1=0,
            i2=4,
            residue_positions=[(0.0, 0.0)],
            bond_length=0.5,
        )


def test_nonpositive_bond_length_raises():
    topology = protein_topology(
        n_residues=5,
        domains=[],
    )
    positions = [(float(i), 0.0, 0.0) for i in range(5)]

    with pytest.raises(ValueError, match="bond_length"):
        get_path_properties(
            topology,
            i1=0,
            i2=4,
            residue_positions=positions,
            bond_length=0.0,
        )


def test_nonpositive_segment_length_raises():
    topology = protein_topology(
        n_residues=5,
        domains=[],
    )
    positions = [(float(i), 0.0, 0.0) for i in range(5)]

    with pytest.raises(ValueError, match="segment_length"):
        get_path_properties(
            topology,
            i1=0,
            i2=4,
            residue_positions=positions,
            bond_length=0.5,
            segment_length=0.0,
        )
