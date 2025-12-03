from __future__ import annotations

import mdtraj as md
import numpy as np

from mpipi_lammps_gen.generate_lammps_files import (
    ProteinData,
    __one_to_three__,
    __three_to_one__,
)

# =========================
# ProteinData -> MDTraj
# =========================


def _protein_to_topology(protein: ProteinData) -> md.Topology:
    """Build an MDTraj Topology from ProteinData (single, unnamed chain)."""

    if protein.atom_types is None or protein.atom_xyz is None:
        msg = "ProteinData is missing atom_types or atom_xyz."
        raise ValueError(msg)

    n_res = len(protein.atom_types)

    if protein.sequence_three_letter is not None:
        seq3 = protein.sequence_three_letter
    else:
        seq3 = [__one_to_three__[aa] for aa in protein.sequence_one_letter]

    if len(seq3) != n_res:
        msg = f"sequence length ({len(seq3)}) != number of residues ({n_res})"
        raise ValueError(msg)

    top = md.Topology()
    chain = top.add_chain()  # single chain (MDTraj will call it 'A')

    for res_idx, (resname, atom_names) in enumerate(
        zip(seq3, protein.atom_types, strict=True), start=1
    ):
        residue = top.add_residue(resname, chain, resSeq=res_idx)

        for atom_name in atom_names:
            # crude element from first char of atom name
            element_symbol = atom_name[0].upper()
            try:
                element = md.element.get_by_symbol(element_symbol)
            except KeyError:
                element = md.element.virtual
            top.add_atom(atom_name, element, residue)

    return top


def _protein_coords_to_nm(protein: ProteinData) -> np.ndarray:
    """Flatten residue-wise Å coordinates into (1, n_atoms, 3) array in nm."""
    if protein.atom_xyz is None:
        msg = "ProteinData.atom_xyz is None"
        raise ValueError(msg)

    flat_coords: list[tuple[float, float, float]] = []
    for res_coords in protein.atom_xyz:
        flat_coords.extend(res_coords)

    xyz_ang = np.asarray(flat_coords, dtype=np.float32)
    if xyz_ang.ndim != 2 or xyz_ang.shape[1] != 3:
        msg = f"Unexpected coordinate shape: {xyz_ang.shape}"
        raise ValueError(msg)

    # Å → nm
    xyz_nm = xyz_ang / 10.0
    return xyz_nm[None, :, :]  # add frame dimension


def protein_to_mdtraj(protein: ProteinData) -> md.Trajectory:
    """Create a single-frame MDTraj Trajectory from ProteinData."""
    top = _protein_to_topology(protein)
    xyz = _protein_coords_to_nm(protein)
    return md.Trajectory(xyz=xyz, topology=top)


# =========================
# MDTraj → ProteinData
# =========================


def _traj_coords_to_ang(
    traj: md.Trajectory, frame: int = 0
) -> list[list[tuple[float, float, float]]]:
    """
    Convert MDTraj trajectory coordinates from nm to Å and group by residue.
    Returns list[residue][(x, y, z)].
    """
    if traj.n_frames == 0:
        msg = "Trajectory has zero frames."
        raise ValueError(msg)

    if not (0 <= frame < traj.n_frames):
        msg = f"Frame index {frame} out of range [0, {traj.n_frames})."
        raise IndexError(msg)

    assert traj.xyz is not None
    xyz_nm = traj.xyz[frame]  # (n_atoms, 3) in nm
    xyz_ang = xyz_nm * 10.0  # nm → Å

    assert traj.topology is not None

    # group atoms by residue in topology order
    residue_coords: list[list[tuple[float, float, float]]] = [
        [] for _ in traj.topology.residues
    ]

    for atom in traj.topology.atoms:
        r_index = atom.residue.index
        x, y, z = xyz_ang[atom.index]
        residue_coords[r_index].append((float(x), float(y), float(z)))

    return residue_coords


def _traj_atom_names_by_residue(traj: md.Trajectory) -> list[list[str]]:
    """Return list[residue][atom_name]."""
    assert traj.topology is not None

    atom_types: list[list[str]] = [[] for _ in traj.topology.residues]
    for atom in traj.topology.atoms:
        atom_types[atom.residue.index].append(atom.name)
    return atom_types


def _traj_seq_three_and_one(traj: md.Trajectory) -> tuple[list[str], list[str]]:
    """
    Extract residue names as 3-letter and 1-letter codes.
    Unknown residues become 'X' in the one-letter sequence.
    """
    seq3: list[str] = []
    seq1: list[str] = []

    assert traj.topology is not None

    for residue in traj.topology.residues:
        r3 = residue.name.upper()
        seq3.append(r3)
        seq1.append(__three_to_one__.get(r3, "X"))

    return seq3, seq1


def _traj_bfactors_to_plddt(traj: md.Trajectory, frame: int = 0) -> list[float] | None:
    """
    Convert per-atom B-factors back to residue-wise pLDDT
    by averaging B-factors of atoms in each residue.

    Returns list[residues] or None if B-factors are absent.
    """
    bf = getattr(traj, "bfactors", None)
    if bf is None:
        # try the attribute used by MDTraj when loading PDBs with B-factors
        bf = getattr(traj, "_bfactors", None)

    if bf is None:
        return None

    if bf.ndim == 2:
        # (n_frames, n_atoms)
        b_frame = bf[frame]
    elif bf.ndim == 1:
        # (n_atoms,)
        b_frame = bf
    else:
        msg = f"Unexpected bfactors shape: {bf.shape}"
        raise ValueError(msg)

    assert traj.topology is not None

    # group bfactors by residue
    residue_b: list[list[float]] = [[] for _ in traj.topology.residues]
    for atom in traj.topology.atoms:
        residue_b[atom.residue.index].append(float(b_frame[atom.index]))

    # average per residue
    plddts: list[float] = []
    for b_list in residue_b:
        if len(b_list) == 0:
            plddts.append(0.0)
        else:
            plddts.append(float(np.mean(b_list)))

    return plddts


def plddt_to_bfactor(protein: ProteinData) -> np.ndarray | None:
    """
    Broadcast residue-wise pLDDT to per-atom B-factors.
    Returns (n_atoms,) or None if pLDDT is missing.
    """
    if protein.plddts is None or protein.atom_types is None:
        return None

    if len(protein.plddts) != len(protein.atom_types):
        msg = "Length of pLDDT array does not match number of residues."
        raise ValueError(msg)

    bfactors: list[float] = []
    for res_idx, atoms in enumerate(protein.atom_types):
        b = float(protein.plddts[res_idx])
        bfactors.extend([b] * len(atoms))

    return np.asarray(bfactors, dtype=np.float32)


def mdtraj_to_protein(traj: md.Trajectory, frame: int = 0) -> ProteinData:
    """
    Convert an MDTraj Trajectory back into a ProteinData object.

    - Takes a single frame (default frame 0).
    - Uses residue/atom ordering from the topology.
    - pLDDTs are derived from B-factors if present (mean per residue).
    - PAE is set to None (cannot be inferred from a generic trajectory).
    - residue_positions is left as None; you can compute them later using
      ProteinData.compute_residue_positions().
    """
    atom_xyz = _traj_coords_to_ang(traj, frame=frame)
    atom_types = _traj_atom_names_by_residue(traj)
    seq3, seq1 = _traj_seq_three_and_one(traj)
    plddts = _traj_bfactors_to_plddt(traj, frame=frame)

    return ProteinData(
        atom_xyz=atom_xyz,
        atom_types=atom_types,
        residue_positions=None,
        sequence_one_letter=seq1,
        sequence_three_letter=seq3,
        plddts=plddts,
        pae=None,
    )
