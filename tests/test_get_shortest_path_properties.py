import pytest

from mpipi_lammps_gen.globular_domains import GlobularDomain, protein_topology
from mpipi_lammps_gen.shortest_path_graph import get_path_properties


def test_path_between_two_terminal_idr_residues():
    topology = protein_topology(
        n_residues=5,
        domains=[],
    )

    positions = [(float(i), 0.0, 0.0) for i in range(5)]

    props = get_path_properties(
        topology,
        i1=1,
        i2=3,
        residue_positions=positions,
        bond_length=0.5,
    )

    assert props.path == []
    assert props.edge_path == []
    assert props.start_offset == pytest.approx(0.0)
    assert props.end_offset == pytest.approx(0.0)
    assert props.total_weight == pytest.approx(1.0)
    assert props.fixed_distances == []
    assert props.n_random_segments == pytest.approx(1.0)


def test_path_from_rigid_domain_to_rigid_domain_uses_domain_shortcut_and_idrs():
    # residues:
    # 0 1 | 2 3 4 | 5 6
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(
        n_residues=7,
        domains=[d0],
    )

    positions = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (2.0, 0.0, 0.0),  # 2
        (2.5, 0.0, 0.0),  # 3
        (3.0, 0.0, 0.0),  # 4
        (4.0, 0.0, 0.0),  # 5
        (5.0, 0.0, 0.0),  # 6
    ]

    props = get_path_properties(
        topology,
        i1=0,
        i2=6,
        residue_positions=positions,
        bond_length=0.5,
    )

    assert props.path == [
        ("start", 0),
        ("CD0", 2),
        ("CD0", 4),
        ("end", 6),
    ]
    assert len(props.edge_path) == 3

    # start and end residues are already anchors
    assert props.start_offset == pytest.approx(0.0)
    assert props.end_offset == pytest.approx(0.0)

    # IDRs: 2 residues on each side => 1.0 + 1.0
    assert props.n_random_segments == pytest.approx(2.0)

    # Shortcut inside domain: distance from residue 2 to 4 = 1.0
    assert props.fixed_distances == pytest.approx([1.0])

    assert props.total_weight == pytest.approx(3.0)

    assert props.n1 == "start"
    assert props.n2 == "end"
    assert props.e1 is None
    assert props.e2 is None
    assert props.start_loop is None
    assert props.end_loop is None


def test_path_between_two_residues_inside_same_domain():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(
        n_residues=7,
        domains=[d0],
    )

    positions = [(float(i), 0.0, 0.0) for i in range(7)]

    props = get_path_properties(
        topology,
        i1=2,
        i2=4,
        residue_positions=positions,
        bond_length=0.5,
    )

    # Same rigid domain: direct shortcut, no anchor-graph traversal needed
    assert props.path == []
    assert props.edge_path == []

    assert props.start_offset == pytest.approx(0.0)
    assert props.end_offset == pytest.approx(0.0)

    assert props.n_random_segments == pytest.approx(0.0)
    assert props.fixed_distances == pytest.approx([2.0])
    assert props.total_weight == pytest.approx(2.0)

    assert props.n1 == "CD0"
    assert props.n2 == "CD0"
    assert props.e1 is None
    assert props.e2 is None
    assert props.start_loop is None
    assert props.end_loop is None


def test_start_residue_inside_self_loop_records_start_loop_metadata():
    # residues:
    # 0 | 1 2 | 3 4 | 5 6 | 7
    d0 = GlobularDomain(indices=[(1, 2), (5, 6)])
    topology = protein_topology(
        n_residues=8,
        domains=[d0],
    )

    positions = [(float(i), 0.0, 0.0) for i in range(8)]

    props = get_path_properties(
        topology,
        i1=3,  # inside self-loop IDR 3..4
        i2=7,
        residue_positions=positions,
        bond_length=0.5,
    )

    assert props.e1 is not None
    assert topology.edges[props.e1]["loop"] is True
    assert props.start_loop == (0, 2)
    assert props.end_loop is None

    # residue 3 is one bond away from the left loop anchor at residue 2
    assert props.start_offset == pytest.approx(0.5)

    assert props.path[-1] == ("end", 7)
    assert props.total_weight > 0.0


def test_end_residue_inside_self_loop_records_end_loop_metadata():
    d0 = GlobularDomain(indices=[(1, 2), (5, 6)])
    topology = protein_topology(
        n_residues=8,
        domains=[d0],
    )

    positions = [(float(i), 0.0, 0.0) for i in range(8)]

    props = get_path_properties(
        topology,
        i1=0,
        i2=4,  # inside self-loop IDR 3..4
        residue_positions=positions,
        bond_length=0.5,
    )

    assert props.start_loop is None
    assert props.e2 is not None
    assert topology.edges[props.e2]["loop"] is True
    assert props.end_loop == (1, 2)

    # residue 4 is one bond away from the right loop anchor at residue 5
    assert props.end_offset == pytest.approx(0.5)
    assert props.total_weight > 0.0


def test_segment_length_normalizes_total_weight_and_idr_contribution():
    d0 = GlobularDomain(indices=[(2, 4)])
    topology = protein_topology(
        n_residues=7,
        domains=[d0],
    )

    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
    ]

    props = get_path_properties(
        topology,
        i1=0,
        i2=6,
        residue_positions=positions,
        bond_length=0.5,
        segment_length=0.25,
    )

    # Without normalization total is 3.0, so now it should be 12.0
    assert props.total_weight == pytest.approx(12.0)

    # Only IDR contributions are accumulated in n_random_segments, also normalized
    assert props.n_random_segments == pytest.approx(8.0)

    # fixed_distances remain physical distances, not normalized weights
    assert props.fixed_distances == pytest.approx([1.0])


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
