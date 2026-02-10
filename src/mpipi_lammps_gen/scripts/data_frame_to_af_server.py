import argparse
from pathlib import Path

from mpipi_lammps_gen.util import read_polars, to_af_server_json


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(dest="i", type=Path, help="Path to input data frame")
    cli.add_argument(dest="o", type=Path, help="Path to output fasta")
    cli.add_argument(dest="seq", type=str, help="Name of sequence column")
    cli.add_argument(
        "--name_col", dest="name", default=None, type=str, help="Name of name column"
    )

    args = cli.parse_args()

    df = read_polars(args.i)

    if args.seq not in df.columns:
        msg = f"No column named {args.seq}, available columsn {df.columns}"
        raise Exception(msg)

    if args.name is not None and args.name not in df.columns:
        msg = f"No column named {args.name}, available columsn {df.columns}"
        raise Exception(msg)

    names = None if args.name is None else df[args.name]

    af_json_str = to_af_server_json(df[args.seq], names=names)

    with Path(args.o).open("w") as f:
        f.write(af_json_str)

    print(f"Wrote {len(df)} sequences to {args.o}.")


if __name__ == "__main__":
    main()
