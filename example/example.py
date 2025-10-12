from pathlib import Path
from typing import Any

import render_jinja2

from mpipi_lammps_gen import alpha_fold_query, generate_lammps_files


def main():
    accession = "A0A096LP49"

    cif_text = alpha_fold_query.query_alphafold(accession).cif_text

    protein_data = generate_lammps_files.parse_cif(cif_text)

    globular_domains = generate_lammps_files.decide_globular_domains(
        protein_data.plddts
    )

    lammps_data = generate_lammps_files.generate_lammps_data(
        protein_data, globular_domains
    )

    data_file_str = generate_lammps_files.write_lammps_data_file(lammps_data)

    data_file_path = Path("data_file")
    with data_file_path.open("w") as f:
        f.write(data_file_str)

    group_file_path = Path("groups")
    with group_file_path.open("w") as f:
        f.write(generate_lammps_files.get_lammps_group_script(lammps_data))

    workdir = Path("./workdir")
    template_file = Path("./templates/rg_script.lmp.jinja")

    variables: dict[str, Any] = {
        "input": {"template": template_file},
        "files": {
            "groups": group_file_path.as_posix(),
            "data": data_file_path.as_posix(),
        },
        "params": {
            "temperature": 300,
            "ionic_strength": 150,
            "n_steps": 10000,
            "timestep": 10.0,
        },
    }

    render_jinja2.render_jinja2(
        working_directory=workdir,
        template_file=template_file,
        included_files=variables["files"].values(),
        variables=variables,
        output=Path("script.lmp"),
    )


if __name__ == "__main__":
    main()
