import functools
import logging
import time
from collections.abc import Generator, Iterable
from typing import Any, NamedTuple

import requests

logger = logging.getLogger()

# AlphaFold DB API: returns JSON metadata with model file URLs keyed by UniProt accession
# e.g. https://alphafold.ebi.ac.uk/api/prediction/P69905
ALPHAFOLD_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def get_with_retries(
    url: str, retries: int, backoff_time: int, timeout: int, **kwargs
) -> requests.Response | None:
    for _ in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, **kwargs)
        except Exception:
            logger.warning("Exception in `get`")
            time.sleep(backoff_time)
            continue
        if r.status_code != 200:
            time.sleep(backoff_time)
            continue
        return r

    return None


def parse_plddt_from_cif(cif_text: str) -> list[float]:
    plddt_list = []

    cif_lines = cif_text.split("\n")

    cur_residue = -1

    for line in cif_lines:
        cols = line.split()

        if len(cols) == 0 or cols[0] != "ATOM":
            continue

        residue_number = int(cols[8])
        plddt = float(cols[14])

        # Two small sanity checks
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


def query_url_from_alphafold_data(
    key: str, alpha_fold_data: dict[str, Any], *args, **kwargs
) -> requests.Response | None:
    url = alpha_fold_data.get(key)

    if url is None:
        logger.warning(
            f"Could not get `{key}` key from alpha fold data. Available keys: {alpha_fold_data.keys()}"
        )
    else:
        response = get_with_retries(url, *args, **kwargs)

        if response is None or response.status_code != 200:
            logger.warning(
                f"Did not receive response for `{key}` key from alpha fold data."
            )
        else:
            return response

    return None


def query_alphafold(
    accession: str,
    timeout: int = 10,
    retries: int = 2,
    get_cif: bool = True,
    get_pdb: bool = True,
    get_pae_matrix: bool = True,
    backoff_time: int = 5,
) -> list[AlphaFoldQueryResult] | None:
    """For a single accession, query the alpha fold database and retrieve some information."""

    url = ALPHAFOLD_PREDICTION_URL.format(accession=accession)

    get = functools.partial(
        get_with_retries, retries=retries, timeout=timeout, backoff_time=backoff_time
    )

    try:
        r = get(url)
    except Exception:
        msg = f"Could not query {url}"
        logger.exception(msg)
        return None

    if r is None:
        return None

    query_results = []

    for alpha_fold_data in r.json():
        sequence = alpha_fold_data.get("sequence")

        plddts = None
        pdb_text = None
        cif_text = None
        pae_matrix = None

        if sequence is None:
            logger.warning(
                f"Could not get `sequence` key from alpha fold data. Available keys: {alpha_fold_data.keys()}"
            )
        else:
            if get_cif:
                res = query_url_from_alphafold_data(
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

            if get_pdb:
                res = query_url_from_alphafold_data(
                    "pdbUrl",
                    alpha_fold_data,
                    retries=retries,
                    backoff_time=backoff_time,
                    timeout=timeout,
                )
                pdb_text = res.text if res is not None else None

            if get_pae_matrix:
                res = query_url_from_alphafold_data(
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
                        assert pae_matrix is not None
                        assert len(pae_matrix) == len(sequence)
                        assert len(pae_matrix) == len(sequence)
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


def query_alphafold_bulk(
    accession_list: Iterable[str], **kwargs
) -> Generator[list[AlphaFoldQueryResult] | None]:
    """For a sequence of accessions, return a generator to query the alpha fold database."""

    for a in accession_list:
        yield query_alphafold(a, **kwargs)


if __name__ == "__main__":
    accession = "A0A096LP49"

    res = query_alphafold(accession, get_pae_matrix=True, get_pdb=False, get_cif=False)

    print(res)
