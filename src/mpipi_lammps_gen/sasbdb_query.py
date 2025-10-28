import functools
import logging
from dataclasses import dataclass
from typing import Any

from mpipi_lammps_gen.alpha_fold_query import get_with_retries

logger = logging.getLogger()

SASBDB_UNIPROT_URL = "https://www.sasbdb.org/rest-api/entry/codes/uniprot/{accession}/"
SASBDB_QUERY_URL = (
    "https://www.sasbdb.org/rest-api/entry/summary/?code={sasbdb_id}&format=json"
)


@dataclass
class SASBDBQueryResult:
    sasbdb_id: str
    uniprot_accession: str

    sequence: str
    uniprot_range_first: int
    uniprot_range_last: int

    temperature_k: float | None
    buffer: str

    rg_guinier: float | None
    rg_guinier_err: float | None
    rg_pddf: float | None
    rg_pddf_err: float | None

    project_title: str
    doi: str | None

    sasbdb_data: dict[str, Any] | None


def query_sasbdb(
    accession: str,
    timeout: int = 10,
    retries: int = 2,
    backoff_time: int = 5,
) -> list[SASBDBQueryResult] | None:
    url = SASBDB_UNIPROT_URL.format(accession=accession)

    get = functools.partial(
        get_with_retries, retries=retries, timeout=timeout, backoff_time=backoff_time
    )

    try:
        r = get(url)
    except Exception:
        msg = f"Could not query {url}"
        logger.exception(msg)
        return None

    query_results = []

    if r is None:
        return None

    for entry in r.json():
        sasbdb_id = entry["code"]
        url = SASBDB_QUERY_URL.format(sasbdb_id=sasbdb_id)

        response = get(url)
        if response is None:
            continue

        if response.status_code != 200:
            continue

        sasbdb_entry = response.json()

        experiment_data = sasbdb_entry["experiment"]

        sample_data = experiment_data["sample"]

        mol_data = sample_data["molecule"]
        if len(mol_data) > 1:
            continue

        assert len(mol_data) == 1

        mol = mol_data[0]
        sequence = mol["sequence"]
        range_first = mol["uniprot_range_first"]
        range_last = mol["uniprot_range_last"]

        buffer = sample_data["buffer"]["name"]

        temp_celsius = experiment_data.get("cell_temperature")

        temp_k = temp_celsius + 273.15 if temp_celsius is not None else None

        rg_guinier = sasbdb_entry["guinier_rg"]
        rg_guinier_err = sasbdb_entry["guinier_rg_error"]

        rg_pddf = sasbdb_entry["pddf_rg"]
        rg_pddf_err = sasbdb_entry["pddf_rg_error"]

        project_title = sasbdb_entry["project"]["title"]

        try:
            doi = sasbdb_entry["project"].get("publication")["doi"]
        except Exception:
            doi = None

        query_results.append(
            SASBDBQueryResult(
                sequence=sequence,
                project_title=project_title,
                sasbdb_id=sasbdb_id,
                uniprot_accession=accession,
                uniprot_range_first=range_first,
                uniprot_range_last=range_last,
                temperature_k=temp_k,
                buffer=buffer,
                doi=doi,
                rg_guinier=rg_guinier,
                rg_guinier_err=rg_guinier_err,
                rg_pddf=rg_pddf,
                rg_pddf_err=rg_pddf_err,
                sasbdb_data=sasbdb_entry,
            )
        )

    return query_results


if __name__ == "__main__":
    acc = "B7UM99"
    acc = "A5FB63"
    res = query_sasbdb(acc)
    print(res)
