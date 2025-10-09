from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

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

mass = {"H": 1, "C": 12, "O": 16, "N": 14, "P": 31, "S": 32}


def decide_globular_domains(
    plddts: Iterable[float], threshold: float = 70.0, minimum_domain_length: int = 3
) -> list[tuple[int, int]]:
    res = []

    in_globular_domain = [p > threshold for p in plddts]
    n_res = len(in_globular_domain)

    # Indices where value changes from False to True
    start_indices = [
        i
        for i in range(1, n_res)
        if not in_globular_domain[i - 1] and in_globular_domain[i]
    ]

    # Indices where value changes from True to False
    end_indices = [
        i
        for i in range(n_res - 1)
        if in_globular_domain[i] and not in_globular_domain[i + 1]
    ]

    # Special checks for the first and last index, since we cannot detect them based on changes from False to True
    if in_globular_domain[0]:
        start_indices.insert(0, 0)

    if in_globular_domain[-1]:
        end_indices.append(n_res - 1)

    assert len(start_indices) == len(end_indices)

    res = list(zip(start_indices, end_indices, strict=True))

    # remove domains, which are below the minimum globular domain length
    for idx, pair in enumerate(res):
        if pair[1] - pair[0] < minimum_domain_length:
            res.pop(idx)

    return res


@dataclass
class ResidueInfo:
    idx: int
    xyz: list[tuple[float, float, float]]
    position: tuple[float, float, float]
    atom_types: list[str]
    plddt: float
    one_letter: str
    three_letter: str


@dataclass
class ProteinData:
    atom_xyz: list[list[tuple[float, float, float]]]
    atom_types: list[list[str]]

    plddts: list[float]
    sequence_one_letter: list[str]
    sequence_three_letter: list[str]

    def compute_residue_position(
        self, types: list[str], xyz: list[tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        idx_ca = types.index("CA")
        return xyz[idx_ca]

    def residue_info(self, idx: int):
        return ResidueInfo(
            idx=idx,
            position=self.compute_residue_position(
                self.atom_types[idx], self.atom_xyz[idx]
            ),
            xyz=self.atom_xyz[idx],
            atom_types=self.atom_types[idx],
            plddt=self.plddts[idx],
            one_letter=self.sequence_one_letter[idx],
            three_letter=self.sequence_three_letter[idx],
        )

    def get_residue_positions(self) -> list[tuple[float, float, float]]:
        residue_positions = []

        for xyz_list, types in zip(self.atom_xyz, self.atom_types, strict=False):
            residue_positions.append(self.compute_residue_position(types, xyz_list))

        return residue_positions


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

    return ProteinData(
        atom_xyz=atom_xyz,
        atom_types=atom_types,
        plddts=plddt_list,
        sequence_one_letter=sequence_one_letter_list,
        sequence_three_letter=sequence_three_letter_list,
    )


def parse_cif_from_path(cif_path: Path) -> ProteinData:
    with cif_path.open() as f:
        cif_text = f.read()
        return parse_cif(cif_text)


@dataclass
class LammpsData:
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
    globular_domains: Iterable[tuple[int, int]],
    box_buffer: float = 20.0,
) -> LammpsData:
    n_residues = len(prot_data.sequence_three_letter)

    def check_if_idx_is_in_globular_domain(idx: int) -> bool:
        return any(glob[0] <= idx and glob[1] >= idx for glob in globular_domains)

    # mass info
    mass_section = []

    mass_section.extend(
        LammpsData.MassRow(atom_type=v[0], mass_value=v[2]) for v in AminoID.values()
    )
    mass_section.extend(
        LammpsData.MassRow(atom_type=v[0], mass_value=v[2]) for v in AminoID.values()
    )

    # box limits
    residue_positions = prot_data.get_residue_positions()

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

        is_in_globular_domain = check_if_idx_is_in_globular_domain(idx_residue)

        atom_type = AminoID[res_info.three_letter][0]

        if is_in_globular_domain:
            atom_type += len(AminoID)

        atom_section.append(
            LammpsData.AtomRow(
                atom_id=idx_residue + 1,
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
        first_in_glob = check_if_idx_is_in_globular_domain(idx_residue)
        second_in_glob = check_if_idx_is_in_globular_domain(idx_residue + 1)

        # if both residues are in a globular domain we skip the bond
        if first_in_glob and second_in_glob:
            continue
        # in this case we are either in an IDR or we are connecting a globular domain to an IDR
        bond_section.append(
            LammpsData.BondRow(
                bond_id=bond_id,
                bond_type=1,
                atom_1=idx_residue + 1,
                atom_2=idx_residue + 2,
            )
        )
        bond_id += 1

    # groups
    groups = []
    for idx, domain in enumerate(globular_domains):
        id_pairs = [(domain[0] + 1, domain[1] + 1)]
        groups.append(LammpsData.Group(name=f"CD{idx + 1}", id_pairs=id_pairs))

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

    res += "\n"
    res += "Bonds\n"
    res += "\n"

    for row in lammps_data.bonds:
        res += f"{row.bond_id} {row.bond_type} {row.atom_1} {row.atom_2}\n"

    return res


def get_lammps_group_script(lammps_data: LammpsData) -> str:
    res = ""

    for g in lammps_data.groups:
        res += f"group {g.name} id "
        res += "".join([f" {p[0]}:{p[1]}" for p in g.id_pairs])
        res += "\n"

    res += (
        "group nonrigid subtract all "
        + " ".join([g.name for g in lammps_data.groups])
        + "\n"
    )

    for num, g in enumerate(lammps_data.groups):
        res += f"fix fxnverigid{num} {g.name} rigid/nvt molecule temp ${{T}} ${{T}} 1000.0\n"

    res += "fix fxnve nonrigid nve\n"
    res += "fix fxlange nonrigid langevin ${{T}} ${{T}} 1000.0 32784\n"

    return res
