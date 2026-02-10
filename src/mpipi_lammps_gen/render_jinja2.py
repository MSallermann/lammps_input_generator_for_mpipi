import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


def _wait_for_files(files: Iterable[Path], timeout: float, poll_interval: float):
    started = time.time()
    while not all(f.exists() for f in files):
        time_cur = time.time()
        if time_cur - started > timeout:
            msg = "Timed out while waiting for files."
            raise Exception(msg)
        time.sleep(poll_interval)


def render_jinja2(
    working_directory: Path | str,
    template_file: Path | str,
    included_files: Iterable[Path | str],
    variables: dict,
    output: Path | str,
    delete_workdir: bool = True,
    latency_wait: float = 5.0,
    latency_poll_interval: float = 0.05,
):
    included_files = [Path(f) for f in included_files]
    working_directory = Path(working_directory)
    template_file = Path(template_file)
    output = Path(output)

    # First we create the working directory
    working_directory.mkdir(exist_ok=True, parents=True)

    # then we copy the template file and all included files over
    for _file in [template_file, *included_files]:
        file = Path(_file)
        if file != working_directory / file.name:
            shutil.copy(file, working_directory)

    # In case we are on a system where there might be a certain
    # latency in the file system we wait for all the files to be present
    _wait_for_files(
        [template_file, *included_files],
        timeout=latency_wait,
        poll_interval=latency_poll_interval,
    )

    env = Environment(loader=FileSystemLoader(str(working_directory)), autoescape=True)

    template = env.get_template(template_file.name)

    with output.open("w") as f:
        f.write(template.render(**variables))

    if delete_workdir:
        shutil.rmtree(working_directory)


if __name__ == "__main__":
    variables: dict[str, Any] = {
        "input": {"template": "./rg_script.lmp.jinja"},
        "files": {"groups": "./groups.lmp", "data": "./lammps.data"},
    }

    workdir = Path("./jinjaworkdir")

    render_jinja2(
        working_directory=workdir,
        template_file=Path("rg_script.lmp.jinja"),
        included_files=variables["files"].values(),
        variables=variables,
        output=workdir / "output.lmp",
    )
