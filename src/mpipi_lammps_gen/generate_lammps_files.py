from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from mpipi_lammps_gen.globular_domains import GlobularDomain

# type, sigma, mass
AminoID = {
    "MET": [1, 6.18, 131.2],
    "GLY": [2, 4.5, 57.05],
    "LYS": [3, 6.36, 128.2],
    "THR": [4, 5.62, 101.1],
    "ARG": [5, 6.56, 156.2],
    "ALA": [6, 5.04, 71.08],
    "ASP": [7, 5.58, 115.1],
    "GLU": [8, 5.92, 129.1],
    "TYR": [9, 6.46, 163.2],
    "VAL": [10, 5.86, 99.07],
    "LEU": [11, 6.18, 113.2],
    "GLN": [12, 6.02, 128.1],
    "TRP": [13, 6.78, 186.2],
    "PHE": [14, 6.36, 147.2],
    "SER": [15, 5.18, 87.08],
    "HIS": [16, 6.08, 137.1],
    "ASN": [17, 5.68, 114.1],
    "PRO": [18, 5.56, 97.12],
    "CYS": [19, 5.48, 103.1],
    "ILE": [20, 6.18, 113.2],
}

__three_to_one__ = {
    "GLY": "G",
    "ALA": "A",
    "VAL": "V",
    "LEU": "L",
    "ILE": "I",
    "THR": "T",
    "SER": "S",
    "MET": "M",
    "CYS": "C",
    "PRO": "P",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
    "HIS": "H",
    "LYS": "K",
    "ARG": "R",
    "ASP": "D",
    "GLU": "E",
    "ASN": "N",
    "GLN": "Q",
}

__one_to_three__ = {
    "G": "GLY",
    "A": "ALA",
    "V": "VAL",
    "L": "LEU",
    "I": "ILE",
    "T": "THR",
    "S": "SER",
    "M": "MET",
    "C": "CYS",
    "P": "PRO",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP",
    "H": "HIS",
    "K": "LYS",
    "R": "ARG",
    "D": "ASP",
    "E": "GLU",
    "N": "ASN",
    "Q": "GLN",
}

mass = {"H": 1, "C": 12, "O": 16, "N": 14, "P": 31, "S": 32}


@dataclass
class ResidueInfo:
    idx: int
    position: tuple[float, float, float]
    one_letter: str
    three_letter: str | None
    xyz: list[tuple[float, float, float]] | None
    atom_types: list[str] | None
    plddt: float | None


@dataclass
class ProteinData:
    atom_xyz: list[list[tuple[float, float, float]]] | None
    atom_types: list[list[str]] | None

    residue_positions: list[tuple[float, float, float]] | None

    sequence_one_letter: list[str]
    sequence_three_letter: list[str] | None

    plddts: list[float] | None
    pae: list[list[float]] | None

    def compute_residue_position(
        self, types: list[str], xyz: list[tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        idx_ca = types.index("CA")
        return xyz[idx_ca]

    def compute_residue_positions(self) -> list[tuple[float, float, float]] | None:
        if self.atom_types is not None and self.atom_xyz is not None:
            self.residue_positions = [
                self.compute_residue_position(types, xyz_list)
                for types, xyz_list in zip(self.atom_types, self.atom_xyz, strict=True)
            ]
        return self.residue_positions

    def residue_info(self, idx: int) -> ResidueInfo | None:
        pos: tuple[float, float, float]

        # Get the position, either from residue positions or by computing it
        if self.residue_positions is not None:
            pos = self.residue_positions[idx]
        elif self.atom_xyz is not None and self.atom_types is not None:
            pos = self.compute_residue_position(
                self.atom_types[idx], self.atom_xyz[idx]
            )
        else:
            return None

        xyz = None if self.atom_xyz is None else self.atom_xyz[idx]
        atom_types = None if self.atom_types is None else self.atom_types[idx]
        three_letter = (
            None
            if self.sequence_three_letter is None
            else self.sequence_three_letter[idx]
        )

        return ResidueInfo(
            idx=idx,
            position=pos,
            xyz=xyz,
            atom_types=atom_types,
            plddt=None if self.plddts is None else self.plddts[idx],  # type: ignore
            one_letter=self.sequence_one_letter[idx],
            three_letter=three_letter,
        )

    def get_residue_positions(self) -> list[tuple[float, float, float]] | None:
        if self.residue_positions is not None:
            return self.residue_positions

        if (
            self.residue_positions is None
            and self.atom_xyz is not None
            and self.atom_types is not None
        ):
            return self.compute_residue_positions()

        return None


def trim_protein(prot: ProteinData, start: int, end: int) -> ProteinData:
    pae = None if prot.pae is None else [row[start:end] for row in prot.pae[start:end]]

    n_residues = len(prot.sequence_one_letter)

    assert start >= 0
    assert start < n_residues
    assert end >= 0
    assert start < n_residues
    assert start < end

    return ProteinData(
        atom_xyz=None if prot.atom_xyz is None else prot.atom_xyz[start:end],
        atom_types=None if prot.atom_types is None else prot.atom_types[start:end],
        residue_positions=(
            None
            if prot.residue_positions is None
            else prot.residue_positions[start:end]
        ),
        pae=pae,
        plddts=None if prot.plddts is None else prot.plddts[start:end],
        sequence_one_letter=prot.sequence_one_letter[start:end],
        sequence_three_letter=(
            None
            if prot.sequence_three_letter is None
            else prot.sequence_three_letter[start:end]
        ),
    )


def parse_cif(cif_text: str) -> ProteinData:
    plddt_list = []

    atom_xyz = []
    atom_types = []

    sequence_one_letter_list = []
    sequence_three_letter_list = []

    cif_lines = cif_text.split("\n")

    cur_residue = -1

    for line in cif_lines:
        cols = line.split()

        if len(cols) == 0 or cols[0] != "ATOM":
            continue

        atom_type = str(cols[3])
        residue_number = int(cols[8])
        plddt = float(cols[14])
        residue_three_letter = str(cols[17])
        residue_one_letter = str(cols[-1])

        # These appends only happen if we "discover" a new residue
        if residue_number != cur_residue:
            atom_xyz.append([])
            atom_types.append([])

            cur_residue = residue_number
            plddt_list.append(plddt)
            sequence_three_letter_list.append(residue_three_letter)
            sequence_one_letter_list.append(residue_one_letter)

        # This always needs to happen
        x, y, z = float(cols[10]), float(cols[11]), float(cols[12])
        atom_xyz[-1].append((x, y, z))
        atom_types[-1].append(atom_type)

        # Two small sanity checks
        if residue_number < 1:
            msg = "Parsed a residue number which is smaller than 1"
            raise Exception(msg)

        if plddt < 0.0 or plddt > 100.0:
            msg = "Parsed a plddt which is not between 1 and 100"
            raise Exception(msg)

    res = ProteinData(
        atom_xyz=atom_xyz,
        atom_types=atom_types,
        residue_positions=None,
        pae=None,
        plddts=plddt_list,
        sequence_one_letter=sequence_one_letter_list,
        sequence_three_letter=sequence_three_letter_list,
    )
    res.compute_residue_positions()
    return res


def parse_cif_from_path(cif_path: Path) -> ProteinData:
    with cif_path.open() as f:
        cif_text = f.read()
        return parse_cif(cif_text)


@dataclass
class LammpsData:
    """Dataclass, which closely represents a LAMMPS data file. All indices count from 1."""

    class AtomRow(NamedTuple):
        atom_id: int
        molecule_tag: int
        atom_type: int
        q: float
        x: float
        y: float
        z: float

    class BondRow(NamedTuple):
        bond_id: int
        bond_type: int
        atom_1: int
        atom_2: int

    class MassRow(NamedTuple):
        atom_type: int
        mass_value: float

    class Group(NamedTuple):
        name: str
        id_pairs: list[tuple[int, int]]

    atoms: list[AtomRow]
    bonds: list[BondRow]
    masses: list[MassRow]
    groups: list[Group]

    x_lims: tuple[float, float]
    y_lims: tuple[float, float]
    z_lims: tuple[float, float]


def generate_lammps_data(
    prot_data: ProteinData,
    globular_domains: Iterable[GlobularDomain],
    box_buffer: float = 20.0,
) -> LammpsData:
    if prot_data.sequence_three_letter is None:
        prot_data.sequence_three_letter = [
            __one_to_three__[r] for r in prot_data.sequence_one_letter
        ]

    n_residues = len(prot_data.sequence_three_letter)

    def check_if_idx_is_rigid(idx: int) -> bool:
        return any(glob.is_in_rigid_region(idx) for glob in globular_domains)

    # mass info
    mass_section = []

    mass_section.extend(
        LammpsData.MassRow(atom_type=v[0], mass_value=v[2]) for v in AminoID.values()
    )
    mass_section.extend(
        LammpsData.MassRow(atom_type=v[0] + len(AminoID), mass_value=v[2])
        for v in AminoID.values()
    )

    # box limits
    residue_positions = prot_data.get_residue_positions()
    assert residue_positions is not None

    x_coords = [r[0] for r in residue_positions]
    y_coords = [r[1] for r in residue_positions]
    z_coords = [r[2] for r in residue_positions]

    x_lo = np.min(x_coords) - box_buffer
    x_hi = np.max(x_coords) + box_buffer

    y_lo = np.min(y_coords) - box_buffer
    y_hi = np.max(y_coords) + box_buffer

    z_lo = np.min(z_coords) - box_buffer
    z_hi = np.max(z_coords) + box_buffer

    # Fill in the atom section
    atom_section = []
    for idx_residue in range(n_residues):
        res_info = prot_data.residue_info(idx_residue)

        # asserts for static type checker
        assert res_info is not None
        assert res_info.three_letter is not None

        is_rigid = check_if_idx_is_rigid(idx_residue)

        atom_type = AminoID[res_info.three_letter][0]

        if is_rigid:
            atom_type += len(AminoID)

        atom_section.append(
            LammpsData.AtomRow(
                atom_id=idx_residue + 1,  # remember to increment indices...
                # remember to increment indices...
                molecule_tag=1,
                atom_type=atom_type,
                q=0.0,
                x=res_info.position[0],
                y=res_info.position[1],
                z=res_info.position[2],
            )
        )

    # Compute bond info
    bond_section = []
    bond_id = 1

    # Within globular domains, bonds are skipped
    for idx_residue in range(n_residues - 1):
        first_rigid = check_if_idx_is_rigid(idx_residue)
        second_rigid = check_if_idx_is_rigid(idx_residue + 1)

        # if both residues are rigid, we skip the bond
        if first_rigid and second_rigid:
            continue

        # in this case we are either in an IDR or we are connecting a globular domain to an IDR
        bond_section.append(
            LammpsData.BondRow(
                bond_id=bond_id,
                bond_type=1,
                atom_1=idx_residue + 1,  # remember to increment indices...
                atom_2=idx_residue + 2,  # remember to increment indices...
            )
        )
        bond_id += 1

    # groups
    groups = []
    for idx, domain in enumerate(globular_domains):
        groups.append(
            LammpsData.Group(name=f"CD{idx + 1}", id_pairs=domain.to_lammps_indices())
        )

    return LammpsData(
        atoms=atom_section,
        bonds=bond_section,
        masses=mass_section,
        groups=groups,
        x_lims=(x_lo, x_hi),
        y_lims=(y_lo, y_hi),
        z_lims=(z_lo, z_hi),
    )


def write_lammps_data_file(lammps_data: LammpsData) -> str:
    res = "Lammps data file\n\n"

    res += f"{len(lammps_data.atoms)} atoms\n"
    res += f"{2 * len(AminoID)} atom types\n"
    res += f"{len(lammps_data.bonds)} bonds\n"
    res += f"{1} bond types\n"
    res += "0 angles\n"
    res += "0 angle types\n"
    res += "0 dihedrals\n"
    res += "0 dihedral types\n"

    res += "\n"
    res += f"{lammps_data.x_lims[0]} {lammps_data.x_lims[1]} xlo xhi\n"
    res += f"{lammps_data.y_lims[0]} {lammps_data.y_lims[1]} ylo yhi\n"
    res += f"{lammps_data.z_lims[0]} {lammps_data.z_lims[1]} zlo zhi\n"
    res += "\n"

    res += "Masses\n\n"

    for row in lammps_data.masses:
        res += f"{row.atom_type} {row.mass_value}\n"

    res += "\n"
    res += "Atoms\n"
    res += "\n"

    for row in lammps_data.atoms:
        res += f"{row.atom_id} {row.molecule_tag} {row.atom_type} {row.q} {row.x} {row.y} {row.z}\n"

    if len(lammps_data.bonds) > 0:
        res += "\n"
        res += "Bonds\n"
        res += "\n"

        for row in lammps_data.bonds:
            res += f"{row.bond_id} {row.bond_type} {row.atom_1} {row.atom_2}\n"

    return res


def get_lammps_group_definition(lammps_data: LammpsData) -> str:
    res = ""

    # define the groups for the globular domains
    for g in lammps_data.groups:
        res += f"group {g.name} id "
        res += "".join([f" {p[0]}:{p[1]}" for p in g.id_pairs])
        res += "\n"

    # define the nonrigid group
    if len(lammps_data.groups) > 0:
        res += (
            "group nonrigid subtract all "
            + " ".join([g.name for g in lammps_data.groups])
            + "\n"
        )
    else:
        res += "group nonrigid union all"

    # neigh modify to exclude intra molecule interaction
    for g in lammps_data.groups:
        res += f"neigh_modify exclude molecule/intra {g.name}\n"

    return res


def get_lammps_minimize_command(
    lammps_data: LammpsData,
    etol: float,
    ftol: float,
    maxiter: int,
    max_eval: int,
    timestep: float,
):
    res = f"# Minimizing with frozen rigid groups with timestep {timestep}\n"

    # fix
    for num, g in enumerate(lammps_data.groups):
        res += f"fix freeze{num} {g.name} setforce 0.0 0.0 0.0\n"

    # run
    res += "min_style cg\n"
    res += f"timestep {timestep}\n"
    res += f"minimize {etol} {ftol} {maxiter} {max_eval}\n"

    # unfix
    for num, _ in enumerate(lammps_data.groups):
        res += f"unfix freeze{num}\n"

    return res


def get_lammps_viscous_command(
    lammps_data: LammpsData,
    n_time_steps: int,
    timestep: float,
    damp: float = 10000.0,
    limit: float = 0.01,
):
    res = "# Minimizing via viscous damping\n"

    # fix
    for num, g in enumerate(lammps_data.groups):
        res += f"fix fxnverigid{num} {g.name} rigid/nve molecule\n"

    res += f"fix fxnve nonrigid nve/limit {limit}\n"
    res += f"fix damp all viscous {damp}\n"
    res += f"timestep {timestep}\n"
    res += f"run {n_time_steps}\n"

    for num, _ in enumerate(lammps_data.groups):
        res += f"unfix fxnverigid{num}\n"

    res += "unfix fxnve\n"
    res += "unfix damp\n"

    return res


def get_lammps_nvt_command(
    lammps_data: LammpsData,
    timestep: float,
    temp: float,
    n_time_steps: int,
    dt_ramp_up: list[float] | None = None,
    lange_damp: float = 1000.0,
    steps_per_stage: int = 10000,
    seed: int = 34278,
) -> str:
    if dt_ramp_up is None:
        dt_ramp_up = []

    res = "# Running NVT ... \n"

    # fix
    for num, g in enumerate(lammps_data.groups):
        res += f"fix fxnverigid{num} {g.name} rigid/nvt molecule temp {temp} {temp} 1000.0\n"

    res += "fix fxnve nonrigid nve\n"
    res += f"fix fxlange nonrigid langevin {temp} {temp} {lange_damp} {seed}\n"

    for i, dt in enumerate(dt_ramp_up):
        res += f"# ... ramping up time step from {dt_ramp_up[0]:.3f} to {dt_ramp_up[-1]:.3f}. Stage {i + 1} / {len(dt_ramp_up)}\n"
        res += f"timestep {dt:.3f}\n"
        res += f"run {steps_per_stage}\n"
        res += f"velocity all create {temp} {seed}\n"

    res += f"# ... running with final timestep of {timestep}\n"
    # run
    res += f"timestep {timestep}\n"
    res += f"run {n_time_steps}\n"

    # unfix
    for num, _ in enumerate(lammps_data.groups):
        res += f"unfix fxnverigid{num}\n"

    res += "unfix fxnve\n"
    res += "unfix fxlange\n"

    return res
