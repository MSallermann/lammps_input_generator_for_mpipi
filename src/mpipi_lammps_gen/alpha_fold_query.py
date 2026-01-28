import asyncio
import logging
from collections.abc import AsyncGenerator, Iterable
from typing import Any, NamedTuple

import httpx

logger = logging.getLogger()

# AlphaFold DB API: returns JSON metadata with model file URLs keyed by UniProt accession
# e.g. https://alphafold.ebi.ac.uk/api/prediction/P69905
ALPHAFOLD_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


async def get_with_retries_async(
    client: httpx.AsyncClient,
    url: str,
    *,
    retries: int,
    backoff_time: float,
    timeout: float,
    **kwargs,
) -> httpx.Response | None:
    """
    Async version of get_with_retries using httpx.

    Retries on:
      - network/timeouts (httpx.RequestError, httpx.TimeoutException)
      - non-200 status codes (keeps parity with your original)
    """
    for attempt in range(retries + 1):
        try:
            r = await client.get(url, timeout=timeout, **kwargs)
        except (httpx.RequestError, httpx.TimeoutException):
            logger.warning("Exception in `get` for %s", url, exc_info=True)
            if attempt < retries:
                await asyncio.sleep(backoff_time)
            continue

        if r.status_code != 200:
            if attempt < retries:
                await asyncio.sleep(backoff_time)
            continue

        return r

    return None


def parse_plddt_from_cif(cif_text: str) -> list[float]:
    plddt_list: list[float] = []
    cif_lines = cif_text.split("\n")
    cur_residue = -1

    for line in cif_lines:
        cols = line.split()
        if len(cols) == 0 or cols[0] != "ATOM":
            continue

        residue_number = int(cols[8])
        plddt = float(cols[14])

        if residue_number < 1:
            msg = "Parsed a residue number which is smaller than 1"
            raise Exception(msg)

        if plddt < 0.0 or plddt > 100.0:
            msg = "Parsed a plddt which is not between 1 and 100"
            raise Exception(msg)

        if residue_number != cur_residue:
            plddt_list.append(plddt)
            cur_residue = residue_number

    return plddt_list


class AlphaFoldQueryResult(NamedTuple):
    accession: str
    alpha_fold_db_id: str
    is_uniprot: bool
    uniprot_start: int
    uniprot_end: int
    sequence: str | None
    cif_text: str | None
    pdb_text: str | None
    plddts: list[float] | None
    pae_matrix: list[list[float]] | None
    alpha_fold_data: dict[str, Any]


async def query_url_from_alphafold_data_async(
    client: httpx.AsyncClient,
    key: str,
    alpha_fold_data: dict[str, Any],
    *,
    retries: int,
    backoff_time: float,
    timeout: float,
) -> httpx.Response | None:
    url = alpha_fold_data.get(key)

    if url is None:
        logger.warning(
            "Could not get `%s` key from alpha fold data. Available keys: %s",
            key,
            list(alpha_fold_data.keys()),
        )
        return None

    response = await get_with_retries_async(
        client,
        url,
        retries=retries,
        backoff_time=backoff_time,
        timeout=timeout,
    )

    if response is None or response.status_code != 200:
        logger.warning(
            "Did not receive response for `%s` key from alpha fold data.", key
        )
        return None

    return response


async def query_alphafold_async(  # noqa: PLR0912
    accession: str,
    *,
    timeout: float = 10.0,
    retries: int = 2,
    get_cif: bool = True,
    get_pdb: bool = True,
    get_pae_matrix: bool = True,
    backoff_time: float = 5.0,
) -> list[AlphaFoldQueryResult] | None:
    """For a single accession, query the AlphaFold DB and retrieve some information."""
    url = ALPHAFOLD_PREDICTION_URL.format(accession=accession)

    async with httpx.AsyncClient(headers={"Accept": "application/json"}) as client:
        try:
            r = await get_with_retries_async(
                client,
                url,
                retries=retries,
                backoff_time=backoff_time,
                timeout=timeout,
            )
        except Exception:
            logger.exception("Could not query %s", url)
            return None

        if r is None:
            return None

        try:
            prediction_list = r.json()
        except Exception:
            logger.exception("Failed to parse JSON from %s", url)
            return None

        query_results: list[AlphaFoldQueryResult] = []

        for alpha_fold_data in prediction_list:
            sequence = alpha_fold_data.get("sequence")

            plddts: list[float] | None = None
            pdb_text: str | None = None
            cif_text: str | None = None
            pae_matrix: list[list[float]] | None = None

            if sequence is None:
                logger.warning(
                    "Could not get `sequence` key from alpha fold data. Available keys: %s",
                    list(alpha_fold_data.keys()),
                )
                continue

            # CIF / pLDDT
            if get_cif:
                res = await query_url_from_alphafold_data_async(
                    client,
                    "cifUrl",
                    alpha_fold_data,
                    retries=retries,
                    backoff_time=backoff_time,
                    timeout=timeout,
                )
                cif_text = res.text if res is not None else None

                if cif_text is not None:
                    plddts = parse_plddt_from_cif(cif_text)
                    if len(sequence) != len(plddts):
                        msg = "sequence and plddts do not have the same length"
                        raise Exception(msg)

            # PDB
            if get_pdb:
                res = await query_url_from_alphafold_data_async(
                    client,
                    "pdbUrl",
                    alpha_fold_data,
                    retries=retries,
                    backoff_time=backoff_time,
                    timeout=timeout,
                )
                pdb_text = res.text if res is not None else None

            # PAE
            if get_pae_matrix:
                res = await query_url_from_alphafold_data_async(
                    client,
                    "paeDocUrl",
                    alpha_fold_data,
                    retries=retries,
                    backoff_time=backoff_time,
                    timeout=timeout,
                )

                try:
                    pae_dict = res.json()[0] if res is not None else None
                    if pae_dict is not None:
                        pae_matrix = pae_dict.get("predicted_aligned_error")
                        if pae_matrix is None:
                            msg = "PAE matrix missing from response"
                            raise Exception(msg)
                        if len(pae_matrix) != len(sequence):
                            msg = "PAE matrix size does not match sequence length"
                            raise Exception(msg)
                except Exception:
                    logger.exception("Exception when trying to parse PAE")

            query_results.append(
                AlphaFoldQueryResult(
                    accession=accession,
                    alpha_fold_db_id=alpha_fold_data["modelEntityId"],
                    is_uniprot=alpha_fold_data["isUniProt"],
                    uniprot_start=alpha_fold_data["uniprotStart"],
                    uniprot_end=alpha_fold_data["uniprotEnd"],
                    sequence=sequence,
                    plddts=plddts,
                    pdb_text=pdb_text,
                    cif_text=cif_text,
                    pae_matrix=pae_matrix,
                    alpha_fold_data=alpha_fold_data,
                )
            )

        return query_results


async def query_alphafold_bulk_async(
    accession_list: Iterable[str],
    **kwargs,
) -> AsyncGenerator[list[AlphaFoldQueryResult] | None, None]:
    """Async generator: for a sequence of accessions, yields results one-by-one."""
    for a in accession_list:
        yield await query_alphafold_async(a, **kwargs)


if __name__ == "__main__":

    async def main() -> None:
        accession = "A0A096LP49"
        res = await query_alphafold_async(
            accession,
            get_pae_matrix=True,
            get_pdb=False,
            get_cif=False,
        )
        print(res)

    asyncio.run(main())
