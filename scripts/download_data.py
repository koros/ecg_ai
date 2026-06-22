#!/usr/bin/env python

import argparse
from pathlib import Path

import wfdb

SMALL_RECORDS = [
    "00",
    "01",
    "03",
]

if args.mode == "small":
    DATABASE = "ltafdb-small"
else args.mode == "full":
    DATABASE = "ltafdb"


def download_subset(output_dir: Path):
    print(f"Downloading subset: {SMALL_RECORDS}")
    wfdb.dl_database(
        DATABASE,
        str(output_dir),
        records=SMALL_RECORDS,
    )


def download_full(output_dir: Path):
    print("Downloading complete database")

    wfdb.dl_database(
        DATABASE,
        str(output_dir),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["small", "full"],
        default="small",
    )

    parser.add_argument(
        "--output-dir",
        default="data/raw",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)/DATABASE

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "small":
        download_subset(output_dir)

    elif args.mode == "full":
        download_full(output_dir)


if __name__ == "__main__":
    main()
