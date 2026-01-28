import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SASBDB_UNIPROT_URL = "https://www.sasbdb.org/rest-api/entry/codes/uniprot/{accession}/"
SASBDB_QUERY_URL = (
    "https://www.sasbdb.org/rest-api/entry/summary/?code={sasbdb_id}&format=json"
)

SASBDB_CODE_RE = re.compile(r"^SAS[A-Z0-9]{3,}$", re.IGNORECASE)


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


async def get_with_retries_async(
    client: httpx.AsyncClient,
    url: str,
    *,
    retries: int,
    backoff_time: float,
    timeout: float,
) -> httpx.Response | None:
    for attempt in range(retries + 1):
        try:
            r = await client.get(url, timeout=timeout)
        except (httpx.RequestError, httpx.TimeoutException):
            if attempt < retries:
                await asyncio.sleep(backoff_time)
            continue

        if r.status_code == 404:
            return None

        if r.status_code != 200:
            if attempt < retries:
                await asyncio.sleep(backoff_time)
            continue

        return r
    return None


async def _fetch_sasbdb_summary(
    client: httpx.AsyncClient,
    sasbdb_id: str,
    uniprot_accession: str,
    *,
    timeout: float,
    retries: int,
    backoff_time: float,
) -> SASBDBQueryResult | None:
    url = SASBDB_QUERY_URL.format(sasbdb_id=sasbdb_id)
    resp = await get_with_retries_async(
        client, url, timeout=timeout, retries=retries, backoff_time=backoff_time
    )
    if resp is None:
        return None

    sasbdb_entry = resp.json()
    experiment_data = sasbdb_entry["experiment"]
    sample_data = experiment_data["sample"]

    sasbdb_entry["experiment"]["sample"]["molecule"][0]

    mol_data = sample_data["molecule"]
    if len(mol_data) != 1:
        return None

    mol = mol_data[0]
    sequence = mol["sequence"]

    if uniprot_accession is None or uniprot_accession == "":
        uniprot_accession = mol["uniprot_code"]
    range_first = mol["uniprot_range_first"]
    range_last = mol["uniprot_range_last"]

    buffer = sample_data["buffer"]["name"]
    temp_celsius = experiment_data.get("cell_temperature")
    temp_k = temp_celsius + 273.15 if temp_celsius is not None else None

    rg_guinier = sasbdb_entry.get("guinier_rg")
    rg_guinier_err = sasbdb_entry.get("guinier_rg_error")
    rg_pddf = sasbdb_entry.get("pddf_rg")
    rg_pddf_err = sasbdb_entry.get("pddf_rg_error")

    project_title = sasbdb_entry["project"]["title"]

    try:
        pub = sasbdb_entry["project"].get("publication")
        doi = pub.get("doi") if isinstance(pub, dict) else None
    except Exception:
        doi = None

    return SASBDBQueryResult(
        sasbdb_id=sasbdb_id,
        uniprot_accession=uniprot_accession,
        sequence=sequence,
        uniprot_range_first=range_first,
        uniprot_range_last=range_last,
        temperature_k=temp_k,
        buffer=buffer,
        rg_guinier=rg_guinier,
        rg_guinier_err=rg_guinier_err,
        rg_pddf=rg_pddf,
        rg_pddf_err=rg_pddf_err,
        project_title=project_title,
        doi=doi,
        sasbdb_data=sasbdb_entry,
    )


async def query_sasbdb_async(
    accession_or_code: str,
    *,
    timeout: float = 10.0,
    retries: int = 2,
    backoff_time: float = 5.0,
    max_concurrency: int = 10,
) -> list[SASBDBQueryResult] | None:
    """
    Accepts either:
      - UniProt accession (uses /codes/uniprot/)
      - SASBDB code (e.g. SASDAT4) (queries summary directly)
    """
    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
    )

    async with httpx.AsyncClient(
        limits=limits, headers={"Accept": "application/json"}
    ) as client:
        # If it's already a SASBDB code, skip the UniProt lookup
        if SASBDB_CODE_RE.match(accession_or_code):
            one = await _fetch_sasbdb_summary(
                client,
                sasbdb_id=accession_or_code,
                uniprot_accession="",  # unknown
                timeout=timeout,
                retries=retries,
                backoff_time=backoff_time,
            )
            return [one] if one is not None else []

        # Otherwise assume UniProt accession
        url = SASBDB_UNIPROT_URL.format(accession=accession_or_code)
        r = await get_with_retries_async(
            client, url, timeout=timeout, retries=retries, backoff_time=backoff_time
        )
        if r is None:
            return None

        codes = r.json()
        sasbdb_ids = [entry["code"] for entry in codes]

        sem = asyncio.Semaphore(max_concurrency)

        async def one_code(sid: str) -> SASBDBQueryResult | None:
            async with sem:
                return await _fetch_sasbdb_summary(
                    client,
                    sasbdb_id=sid,
                    uniprot_accession=accession_or_code,
                    timeout=timeout,
                    retries=retries,
                    backoff_time=backoff_time,
                )

        results = await asyncio.gather(*(one_code(sid) for sid in sasbdb_ids))
        return [r for r in results if r is not None]
