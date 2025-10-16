import shutil
from pathlib import Path
from typing import Any

import polars as pl

from mpipi_lammps_gen.generate_lammps_files import (
    decide_globular_domains,
    generate_lammps_data,
    get_lammps_group_definition,
    get_lammps_minimize_command,
    get_lammps_nvt_command,
    parse_cif,
    write_lammps_data_file,
)
from mpipi_lammps_gen.render_jinja2 import render_jinja2

df = pl.read_parquet("query_results_sv.parquet")[88]

temp = 293.0
ionic_strength = 150.0
n_steps = 10000
timestep = 10.0

threshold = 40.0
minimum_domain_length = 3
minimum_idr_length = 3


for accession, plddts, cif_text in zip(
    df["accession"], df["plddts"], df["cif_text"], strict=True
):
    output = Path(f"./output/{accession}")
    output.mkdir(parents=True, exist_ok=True)

    data_file_path = output / "data_file.data"

    workdir = output / "./workdir"
    workdir.mkdir(exist_ok=True)

    script_path = output / "script.lmp"
    template_file = Path("./templates/rg_script.lmp.jinja")

    protein_data = parse_cif(cif_text=cif_text)

    globular_domains = decide_globular_domains(
        plddts=plddts,
        threshold=threshold,
        minimum_domain_length=minimum_domain_length,
        minimum_idr_length=minimum_idr_length,
    )

    lammps_data = generate_lammps_data(protein_data, globular_domains)

    data_file_str = write_lammps_data_file(lammps_data)
    with data_file_path.open("w") as f:
        f.write(data_file_str)

    groups_definition_str = get_lammps_group_definition(lammps_data)

    min_cmd = get_lammps_minimize_command(
        lammps_data, etol=0.0, ftol=0.0023, maxiter=10000, max_eval=40000, timestep=0.1
    )

    nvt_cmd = get_lammps_nvt_command(
        lammps_data, timestep=timestep, temp=temp, n_time_steps=n_steps
    )

    variables: dict[str, Any] = {
        "input": {"template": template_file},
        "params": {
            "temperature": temp,
            "n_steps": n_steps,
            "ionic_strength": ionic_strength,
            "groups_definition_str": groups_definition_str,
            "data_file_name": data_file_path.name,
            "run_str": min_cmd,
        },
    }

    render_jinja2(
        working_directory=workdir,
        template_file=template_file,
        included_files=[],
        variables=variables,
        output=script_path,
    )

    shutil.rmtree(workdir)
