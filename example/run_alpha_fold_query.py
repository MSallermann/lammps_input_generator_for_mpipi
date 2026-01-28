import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from mpipi_lammps_gen.alpha_fold_query import query_alphafold_async

logger = logging.getLogger(__name__)


def find_next_free_file(file: Path, n_max: int = 1000) -> Path:
    test_name = Path()

    for i in range(1, n_max + 1):
        test_name = file.with_name(file.with_suffix("").name + f"_{i}").with_suffix(
            file.suffix
        )
        if not test_name.exists():
            return test_name

    msg = (
        f"Could not find free file within {n_max} attempts. Last attempt `{test_name}`"
    )
    raise Exception(msg)


def save(query_results: list[Any], output_path: Path) -> None:
    if not query_results:
        return

    df_out_new = pl.DataFrame(query_results)
    if len(df_out_new) == 0:
        return

    if output_path.exists():
        logger.info("`%s` exists. Trying to read as parquet.", output_path)
        try:
            df_out_old = pl.read_ipc(output_path)
            logger.info("... success! There are %d rows in the df", len(df_out_old))
        except Exception:
            df_out_old = None
            logger.exception("... failed!", stack_info=True)
    else:
        df_out_old = None

    if df_out_old is None:
        df_out = df_out_new
    else:
        logger.info("Trying to concat dataframes.")
        try:
            df_out = pl.concat((df_out_old, df_out_new), how="vertical_relaxed")
            logger.info("... success!")
        except Exception:
            df_out = df_out_new
            output_path = find_next_free_file(output_path)
            logger.exception(
                "Cannot concatenate data frames. Changing output path to not overwrite data"
            )

    logger.info("Saving %d rows to `%s`", len(df_out), output_path)
    df_out.write_ipc(output_path)


async def main_async(
    accessions: Iterable[str],
    output_path: Path,
    *,
    n_flush: int = 100,
    max_concurrency: int = 20,
    timeout: float = 10.0,
    retries: int = 1,
    backoff_time: float = 1.0,
    # pass-through toggles to query_alphafold_async if you want
    get_cif: bool = True,
    get_pdb: bool = True,
    get_pae_matrix: bool = True,
) -> None:
    """
    Async driver around query_alphafold_async.

    - Runs accession queries concurrently (bounded by max_concurrency)
    - Each accession can yield multiple AF models; we write ONE ROW PER MODEL
    - Flushes accumulated rows to parquet every n_flush completed accessions
    """
    accessions_list = list(accessions)
    total = len(accessions_list)

    query_results: list[Any] = []
    completed = 0

    sem = asyncio.Semaphore(max_concurrency)

    async def one(acc: str) -> list[dict[str, Any]]:
        nonlocal completed
        async with sem:
            models = await query_alphafold_async(
                acc,
                timeout=timeout,
                retries=retries,
                backoff_time=backoff_time,
                get_cif=get_cif,
                get_pdb=get_pdb,
                get_pae_matrix=get_pae_matrix,
            )

        completed += 1
        logger.info("Queried %s [%d / %d]", acc, completed, total)

        # One row per AF model (NamedTuple -> dict)
        if not models:
            return []
        return [m._asdict() for m in models]

    tasks = [asyncio.create_task(one(a)) for a in accessions_list]

    for fut in asyncio.as_completed(tasks):
        try:
            rows = await fut
        except Exception:
            logger.exception("Query task failed")
            rows = []

        if rows:
            query_results.extend(rows)

        if completed != 0 and completed % n_flush == 0:
            save(query_results=query_results, output_path=output_path)
            query_results = []

    save(query_results=query_results, output_path=output_path)


def main(accessions: Iterable[str], output_path: Path, n_flush: int = 100) -> None:
    asyncio.run(
        main_async(accessions=accessions, output_path=output_path, n_flush=n_flush)
    )


if __name__ == "__main__":
    from logging import FileHandler

    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(), FileHandler("query_alphafold.log")],
    )

    # optional: quiet down noisy httpx request logs if enabled elsewhere
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # ids_to_query = pl.read_parquet("./data_idrs.parquet").sample(8000)["uniprot_id"]

    ids_to_query = pl.read_parquet(
        "/home/sie/Biocondensates/exp_data/1_post_process/peptone.parquet"
    )["uniprot_accession"]
    output_path = Path("./af_db_peptone.arrow")

    main(
        accessions=ids_to_query,
        output_path=output_path,
        n_flush=5000,
    )
