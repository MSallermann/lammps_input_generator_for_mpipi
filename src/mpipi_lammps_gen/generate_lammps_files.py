from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np

from mpipi_lammps_gen.globular_domains import GlobularDomain
from mpipi_lammps_gen.place_proteins import place_proteins_in_grid

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


def is_valid_one_letter_sequence(seq: str | Iterable[str]):
    return all(res in __one_to_three__ for res in seq)


def is_valid_three_letter_sequence(seq: Iterable[str]):
    return all(res in __three_to_one__ for res in seq)


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
        self,
        types: list[str],
        xyz: list[tuple[float, float, float]],
        method: Literal["Ca", "com"] = "Ca",
    ) -> tuple[float, float, float]:
        if method == "Ca":
            idx_ca = types.index("CA")
            return xyz[idx_ca]
        elif method == "com":  # noqa: RET505
            com_position = np.zeros(3)
            total_mass = 0.0

            for pos, t in zip(xyz, types, strict=True):
                m = mass[t[0]]  # We have to look at the first char
                total_mass += m
                com_position += m * np.array(pos)
            return (
                float(com_position[0] / total_mass),
                float(com_position[1] / total_mass),
                float(com_position[2] / total_mass),
            )

        msg = f"Invalid method: {method}"
        raise Exception(msg)

    def compute_residue_positions(
        self,
        method: Literal["Ca", "com"]
        | Iterable[Literal["Ca", "com"]]
        | Iterable[str] = "Ca",
    ) -> list[tuple[float, float, float]] | None:
        if self.atom_types is not None and self.atom_xyz is not None:
            if isinstance(method, str):
                # TODO(MS): e suppress pyright here. Fix it maybe.
                method = len(self.atom_types) * [method]  # pyright: ignore[reportAssignmentType]

            self.residue_positions = [
                self.compute_residue_position(types, xyz_list, meth)  # pyright: ignore[reportArgumentType]
                for types, xyz_list, meth in zip(
                    self.atom_types, self.atom_xyz, method, strict=True
                )
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

    ### check the start/end/the number of residues
    good = start >= 0
    good = good and start < n_residues
    good = good and end >= 0
    good = good and start < n_residues
    good = good and start < end

    if not good:
        msg = f"There is a problem with the start/end indices or the number of residues: {start = }, {end = }, {n_residues = }"
        raise Exception(msg)

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


def parse_cif(
    cif_text: str, method: Literal["Ca", "com"] | Iterable[Literal["Ca", "com"]]
) -> ProteinData:
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
    res.compute_residue_positions(method=method)
    return res


def parse_cif_from_path(
    cif_path: Path, method: Literal["Ca", "com"] = "Ca"
) -> ProteinData:
    with cif_path.open() as f:
        cif_text = f.read()
        return parse_cif(cif_text, method=method)


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
    n_proteins_x: int = 1,
    n_proteins_y: int = 1,
    n_proteins_z: int = 1,
    grid_buffer: float = 6.0,
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

    n_proteins_total = n_proteins_x * n_proteins_y * n_proteins_z

    protein_positions = place_proteins_in_grid(
        residue_positions=residue_positions,
        n_proteins_x=n_proteins_x,
        n_proteins_y=n_proteins_y,
        n_proteins_z=n_proteins_z,
        grid_buffer=grid_buffer,
    )

    protein_positions_arr = np.ravel(np.array(protein_positions))
    protein_positions_arr = protein_positions_arr.reshape(
        len(protein_positions_arr) // 3, 3
    )

    x_coords = protein_positions_arr[:, 0]
    y_coords = protein_positions_arr[:, 1]
    z_coords = protein_positions_arr[:, 2]

    x_lo = np.min(x_coords) - box_buffer
    x_hi = np.max(x_coords) + box_buffer
    y_lo = np.min(y_coords) - box_buffer
    y_hi = np.max(y_coords) + box_buffer
    z_lo = np.min(z_coords) - box_buffer
    z_hi = np.max(z_coords) + box_buffer

    # Fill in the atom section
    atom_section = []
    for idx_protein in range(n_proteins_total):
        for idx_residue in range(n_residues):
            res_info = prot_data.residue_info(idx_residue)

            # asserts for static type checker
            assert res_info is not None
            assert res_info.three_letter is not None

            is_rigid = check_if_idx_is_rigid(idx_residue)

            atom_type = AminoID[res_info.three_letter][0]

            if is_rigid:
                atom_type += len(AminoID)

            # atom_id has to be unique for every atom
            atom_id = (idx_residue + 1) + n_residues * idx_protein

            atom_section.append(
                LammpsData.AtomRow(
                    atom_id=atom_id,  # remember to increment indices...
                    # remember to increment indices...
                    molecule_tag=idx_protein + 1,
                    atom_type=atom_type,
                    q=0.0,
                    x=protein_positions[idx_protein][idx_residue][0],
                    y=protein_positions[idx_protein][idx_residue][1],
                    z=protein_positions[idx_protein][idx_residue][2],
                )
            )

    # Compute bond info
    bond_section = []
    bond_id = 1

    # Within globular domains, bonds are skipped
    for idx_protein in range(n_proteins_total):
        for idx_residue in range(n_residues - 1):
            first_rigid = check_if_idx_is_rigid(idx_residue)
            second_rigid = check_if_idx_is_rigid(idx_residue + 1)

            # if both residues are rigid, we skip the bond
            if first_rigid and second_rigid:
                continue

            atom_id_1 = (idx_residue + 1) + n_residues * idx_protein
            atom_id_2 = (idx_residue + 2) + n_residues * idx_protein

            # in this case we are either in an IDR or we are connecting a globular domain to an IDR
            bond_section.append(
                LammpsData.BondRow(
                    bond_id=bond_id,
                    bond_type=1,
                    atom_1=atom_id_1,  # remember to increment indices...
                    atom_2=atom_id_2,  # remember to increment indices...
                )
            )
            bond_id += 1

    # groups
    groups = []
    for idx_domain, domain in enumerate(globular_domains):
        # We first get the indices for the "first" protein
        id_pairs_single_prot = domain.to_lammps_indices()

        id_pairs = []
        # Then for each protein, we have to add to the groups while offsetting by the number of residues
        for idx_protein in range(n_proteins_total):
            id_pairs.extend(
                [
                    (i + idx_protein * n_residues, j + idx_protein * n_residues)
                    for i, j in id_pairs_single_prot
                ]
            )

        groups.append(LammpsData.Group(name=f"CD{idx_domain + 1}", id_pairs=id_pairs))

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

    n_atom_types = 40  # max(int(a.atom_type) for a in lammps_data.atoms)

    res += f"{len(lammps_data.atoms)} atoms\n"
    res += f"{n_atom_types} atom types\n"
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

    if len(lammps_data.masses) > 0:
        res += "Masses\n\n"
        for row in lammps_data.masses:
            res += f"{row.atom_type:.0f} {row.mass_value}\n"

    if len(lammps_data.atoms) > 0:
        res += "\n"
        res += "Atoms\n"
        res += "\n"

        for row in lammps_data.atoms:
            res += f"{row.atom_id:.0f} {row.molecule_tag:.0f} {row.atom_type:.0f} {row.q} {row.x} {row.y} {row.z}\n"

    if len(lammps_data.bonds) > 0:
        res += "\n"
        res += "Bonds\n"
        res += "\n"

        for row in lammps_data.bonds:
            res += f"{row.bond_id:.0f} {row.bond_type:.0f} {row.atom_1:.0f} {row.atom_2:.0f}\n"

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


def get_lammps_npt_command(
    lammps_data: LammpsData,
    timestep: float,
    temp: float,
    press: float,
    n_time_steps: int,
    dt_ramp_up: list[float] | None = None,
    steps_per_stage: int = 10000,
    pdamp: float = 1000.0,
    lange_damp: float = 1000.0,
    seed: int = 34278,
) -> str:
    if dt_ramp_up is None:
        dt_ramp_up = []

    res = "# Running NPT ... \n"

    for num, g in enumerate(lammps_data.groups):
        res += f"fix fxnverigid{num} {g.name} rigid/nvt molecule temp {temp} {temp} 1000.0\n"

    # barostat only
    res += f"fix fxbaro nonrigid nph iso {press} {press} {pdamp} dilate all\n"

    # langevin thermostat
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

    res += "unfix fxbaro\n"
    res += "unfix fxlange\n"

    return res
