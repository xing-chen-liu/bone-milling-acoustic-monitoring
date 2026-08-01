#!/usr/bin/env python3
"""
Separate the two known sound-source components in the 8.1 CSV dataset.

The mixed files are:
    1.csv ... 6.csv

The known component files are:
    01.csv ... 06.csv  source propagating from microphone 1 to 6
    11.csv ... 16.csv  source propagating from microphone 6 to 1

The script verifies, sample by sample, that:
    mixed mic N = source_1 mic N + source_2 mic N

Then it writes two CSV files. Each output row is one sample; each output
has six columns, corresponding to microphones 1 ... 6.
"""

from __future__ import annotations

import argparse
import csv
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, TextIO


MIXED_FILES = tuple(f"{i}.csv" for i in range(1, 7))
SOURCE_1_FILES = tuple(f"0{i}.csv" for i in range(1, 7))
SOURCE_2_FILES = tuple(f"1{i}.csv" for i in range(1, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate two sound sources from the 8.1 microphone CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default="8.1",
        type=Path,
        help="Directory containing 1-6.csv, 01-06.csv, and 11-16.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory for separated CSV files. Default: <input-dir>/separated",
    )
    parser.add_argument(
        "--tolerance",
        default=1e-8,
        type=float,
        help="Maximum allowed absolute error when checking mixed = source1 + source2.",
    )
    parser.add_argument(
        "--with-header",
        action="store_true",
        help="Write a mic1,mic2,...,mic6 header row to each output CSV.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip sample-by-sample verification of the mixed files.",
    )
    return parser.parse_args()


def require_files(input_dir: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (input_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required CSV file(s) in {input_dir}: {joined}")


def open_readers(input_dir: Path, names: Iterable[str]) -> tuple[list[TextIO], list[csv.reader]]:
    handles: list[TextIO] = []
    readers: list[csv.reader] = []
    for name in names:
        handle = (input_dir / name).open("r", encoding="utf-8-sig", newline="")
        handles.append(handle)
        readers.append(csv.reader(handle))
    return handles, readers


def first_cell(row: list[str], file_name: str, row_number: int) -> str:
    if not row or not row[0].strip():
        raise ValueError(f"{file_name} has an empty value at row {row_number}")
    return row[0].strip()


def close_all(handles: Iterable[TextIO]) -> None:
    for handle in handles:
        handle.close()


def separate_sources(
    input_dir: Path,
    output_dir: Path,
    tolerance: float,
    with_header: bool,
    verify: bool,
) -> tuple[Path, Path, int, float]:
    require_files(input_dir, MIXED_FILES + SOURCE_1_FILES + SOURCE_2_FILES)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_1_out = output_dir / "source_1_1to6.csv"
    source_2_out = output_dir / "source_2_6to1.csv"

    all_names = MIXED_FILES + SOURCE_1_FILES + SOURCE_2_FILES
    input_handles, readers = open_readers(input_dir, all_names)

    max_error = 0.0
    sample_count = 0

    try:
        with source_1_out.open("w", encoding="utf-8", newline="") as s1_file, source_2_out.open(
            "w", encoding="utf-8", newline=""
        ) as s2_file:
            source_1_writer = csv.writer(s1_file, lineterminator="\n")
            source_2_writer = csv.writer(s2_file, lineterminator="\n")

            if with_header:
                header = [f"mic{i}" for i in range(1, 7)]
                source_1_writer.writerow(header)
                source_2_writer.writerow(header)

            for row_number, rows in enumerate(zip_longest(*readers), start=1):
                if any(row is None for row in rows):
                    raise ValueError(
                        "CSV files have different numbers of rows; "
                        f"first mismatch found near row {row_number}."
                    )

                mixed_rows = rows[:6]
                source_1_rows = rows[6:12]
                source_2_rows = rows[12:18]

                mixed = [
                    first_cell(row, MIXED_FILES[index], row_number)
                    for index, row in enumerate(mixed_rows)
                ]
                source_1 = [
                    first_cell(row, SOURCE_1_FILES[index], row_number)
                    for index, row in enumerate(source_1_rows)
                ]
                source_2 = [
                    first_cell(row, SOURCE_2_FILES[index], row_number)
                    for index, row in enumerate(source_2_rows)
                ]

                if verify:
                    for index in range(6):
                        error = abs(float(mixed[index]) - (float(source_1[index]) + float(source_2[index])))
                        max_error = max(max_error, error)
                        if error > tolerance:
                            mic = index + 1
                            raise ValueError(
                                f"Verification failed at row {row_number}, mic {mic}: "
                                f"{mixed[index]} != {source_1[index]} + {source_2[index]} "
                                f"(abs error {error:.3g}, tolerance {tolerance:.3g})"
                            )

                source_1_writer.writerow(source_1)
                source_2_writer.writerow(source_2)
                sample_count += 1
    finally:
        close_all(input_handles)

    return source_1_out, source_2_out, sample_count, max_error


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "separated"

    source_1_out, source_2_out, sample_count, max_error = separate_sources(
        input_dir=input_dir,
        output_dir=output_dir,
        tolerance=args.tolerance,
        with_header=args.with_header,
        verify=not args.no_verify,
    )

    print(f"Wrote {source_1_out}")
    print(f"Wrote {source_2_out}")
    print(f"Samples per file: {sample_count}")
    if not args.no_verify:
        print(f"Max reconstruction error: {max_error:.12g}")


if __name__ == "__main__":
    main()
