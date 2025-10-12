import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


def render_jinja2(
    working_directory: Path,
    template_file: Path,
    included_files: Iterable[Path],
    variables: dict,
    output: Path,
):
    # First we create the working directory
    working_directory.mkdir(exist_ok=True, parents=True)

    # then we copy the template file and all included files over
    for file in [template_file, *included_files]:
        if file != working_directory / file.name:
            shutil.copy(file, working_directory)

    env = Environment(loader=FileSystemLoader(str(working_directory)), autoescape=True)

    template = env.get_template(template_file.name)

    with output.open("w") as f:
        f.write(template.render(**variables))


if __name__ == "__main__":
    variables: dict[str, Any] = {
        "input": {"template": "./rg_script.lmp.jinja"},
        "files": {"groups": "./groups.lmp", "data": "./lammps.data"},
    }

    workdir = Path("./workdir")

    render_jinja2(
        working_directory=workdir,
        template_file=Path("rg_script.lmp.jinja"),
        included_files=variables["files"].values(),
        variables=variables,
        output=workdir / "output.lmp",
    )
