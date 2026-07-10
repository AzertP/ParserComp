#!/usr/bin/env python3
"""
Combine benchmark CSV files from valid and invalid benchmark runs.

Default inputs:
    results/benchmark_csv
    results/benchmark_csv_invalid

Default output:
    results_comprehensive

Invalid files are written under their base benchmark name:
    benchmark_json_invalid.csv -> benchmark_json.csv
"""

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALID_DIR = PROJECT_ROOT / "results" / "benchmark_csv"
DEFAULT_INVALID_DIR = PROJECT_ROOT / "results" / "benchmark_csv_invalid"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_comprehensive"


def output_name_for_invalid(path):
    stem = path.stem
    if stem.endswith("_invalid"):
        stem = stem[: -len("_invalid")]
    return f"{stem}{path.suffix}"


def add_columns(columns, fieldnames):
    for fieldname in fieldnames:
        if fieldname not in columns:
            columns.append(fieldname)


def read_csv(path):
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header")

        rows = []
        for row in reader:
            if None in row:
                raise ValueError(f"{path} has rows with more fields than the header")
            rows.append(row)

    return reader.fieldnames, rows


def add_file(groups, output_name, path):
    fieldnames, rows = read_csv(path)
    group = groups.setdefault(output_name, {"columns": [], "rows": []})
    add_columns(group["columns"], fieldnames)
    group["rows"].extend(rows)


def combine_csvs(valid_dir, invalid_dir, output_dir):
    groups = {}

    if not valid_dir.is_dir():
        raise FileNotFoundError(f"Valid results directory not found: {valid_dir}")
    if not invalid_dir.is_dir():
        raise FileNotFoundError(f"Invalid results directory not found: {invalid_dir}")

    for path in sorted(valid_dir.glob("*.csv")):
        add_file(groups, path.name, path)

    for path in sorted(invalid_dir.glob("*.csv")):
        add_file(groups, output_name_for_invalid(path), path)

    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = 0
    written_rows = 0
    for output_name in sorted(groups):
        group = groups[output_name]
        output_path = output_dir / output_name
        with output_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=group["columns"],
                restval="",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(group["rows"])

        written_files += 1
        written_rows += len(group["rows"])

    return written_files, written_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine valid and invalid benchmark CSVs into comprehensive results."
    )
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=DEFAULT_VALID_DIR,
        help=f"Directory containing valid benchmark CSVs (default: {DEFAULT_VALID_DIR})",
    )
    parser.add_argument(
        "--invalid-dir",
        type=Path,
        default=DEFAULT_INVALID_DIR,
        help=f"Directory containing invalid benchmark CSVs (default: {DEFAULT_INVALID_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for combined CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    written_files, written_rows = combine_csvs(
        args.valid_dir,
        args.invalid_dir,
        args.output_dir,
    )
    print(f"Wrote {written_files} CSV files with {written_rows} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
