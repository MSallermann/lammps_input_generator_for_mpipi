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
) -> requests.Response:
    for _ in range(retries):
        r = requests.get(url, timeout=timeout, **kwargs)
        if r.status_code != 200:
            time.sleep(backoff_time)
            continue
        return r

    return requests.get(url, timeout=timeout, **kwargs)


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
    http_status: int
    accession: str
    sequence: str | None
    cif_text: str | None
    pdb_text: str | None
    plddts: list[float] | None
    pae_matrix: list[list[float]] | None
    alpha_fold_data: dict[str, Any] | None


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

        if response.status_code != 200:
            logger.warning(
                f"Received response code {response.status_code} for request of cif file"
            )
        else:
            return response

    return None


def query_alphafold(  # noqa: PLR0912, PLR0915
    accession: str,
    timeout: int = 10,
    retries: int = 2,
    get_cif: bool = True,
    get_pdb: bool = True,
    get_pae_matrix: bool = True,
    backoff_time: int = 5,
) -> AlphaFoldQueryResult:
    """For a single accession, query the alpha fold database and retrieve some information."""

    url = ALPHAFOLD_PREDICTION_URL.format(accession=accession)

    # These are the return values
    response_code_alpha_fold: int = -1
    sequence: str | None = None
    plddts: list[float] | None = None
    alpha_fold_data: dict[str, Any] | None = None
    cif_text: str | None = None
    pdb_text: str | None = None
    pae_matrix: list[list[float]] | None = None

    try:
        r = get_with_retries(
            url, retries=retries, backoff_time=backoff_time, timeout=timeout
        )

        response_code_alpha_fold = r.status_code

        if response_code_alpha_fold != 200:
            logger.warning(f"Received response code {response_code_alpha_fold}")
        else:
            try:
                alpha_fold_data = r.json()[0]
            except Exception:
                logger.exception("Could not convert response to a single json.")

            if alpha_fold_data is None:
                logger.warning("Received `None` for alpha_fold_data")
            else:
                sequence = alpha_fold_data.get("sequence")

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
    except Exception:
        logger.exception("Exception in `query_alphafold`")
    finally:
        ...

    return AlphaFoldQueryResult(
        http_status=response_code_alpha_fold,
        accession=accession,
        sequence=sequence,
        plddts=plddts,
        pdb_text=pdb_text,
        cif_text=cif_text,
        pae_matrix=pae_matrix,
        alpha_fold_data=alpha_fold_data,
    )


def query_alphafold_bulk(
    accession_list: Iterable[str], **kwargs
) -> Generator[AlphaFoldQueryResult]:
    """For a sequence of accessions, return a generator to query the alpha fold database."""

    for a in accession_list:
        yield query_alphafold(a, **kwargs)


if __name__ == "__main__":
    import numpy as np

    accession = "A0A096LP49"

    res = query_alphafold(accession, get_pae_matrix=True, get_pdb=False, get_cif=False)

    print(np.array(res.pae_matrix))
    print(np.array(res.pae_matrix).shape)

    # print(res.sequence)
    # print(res.plddts)

    # with Path("ex.cif").open("w") as f:
    #     if res.cif_text is not None:
    #         f.write(res.cif_text)

    # print(res.cif_text)
    # print(res.alpha_fold_data)
    # print(res.http_status)

    # id_list = [
    #     "A0A024RBG1",
    #     "A0A075B6T7",
    #     "A0A087WTH1",
    #     "A0A087WTH5",
    # ]

    # res = query_alphafold_bulk(id_list, get_cif=False)
    # print([r.sequence for r in res])
