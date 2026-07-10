#!/usr/bin/env python3
"""Generate invalid scannerless benchmark inputs.

The generator derives candidates from the valid corpora used by
src/bin/benchmark_csv.rs and keeps only candidates rejected by the binary built
from src/main.rs in character-tokenizer mode.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SEED = 20260709
DEFAULT_TARGET_COUNT = 30
DEFAULT_MAX_BYTES = 4096
VALIDATION_TIMEOUT_SECONDS = 5
METHODS = ("cut_prefix", "cut_suffix", "mutate_char")
STRATA_COUNT = 5


@dataclass(frozen=True)
class ScannerlessConfig:
    name: str
    grammar_path: str
    input_path: str
    # Per-grammar cap on source (and therefore candidate) length.  Left at
    # DEFAULT_MAX_BYTES for most grammars; overridden for grammars where
    # scannerless parsing time grows super-polynomially with input length:
    #
    #   CSS  (850):  largest length in benchmark_css.csv where every measured
    #                parser completes in under 1 s.
    #   HTML (967):  same criterion applied to benchmark_html.csv.
    #   shell (1500): benchmark_shell.csv covers complete scripts; prefix/suffix
    #                 cuts exhibit worse complexity.  1500 is the empirically
    #                 verified longest cut that validates within 1 s (release).
    #
    # This aligns the invalid corpus with the performance envelope already
    # characterised by the valid benchmark, keeping every validation call
    # tractable and making the two corpora directly comparable in the paper.
    max_source_bytes: int = DEFAULT_MAX_BYTES

    @property
    def output_path(self) -> str:
        return f"input/{self.name}_invalid.txt"


SCANNERLESS_CONFIGS: tuple[ScannerlessConfig, ...] = (
    ScannerlessConfig("ansi_c", "grammars/ansi_c.json", "input/ansi_c.txt"),
    ScannerlessConfig("bool", "grammars/bool.json", "input/bool.txt"),
    ScannerlessConfig("expr", "grammars/calc.json", "input/expr.txt"),
    ScannerlessConfig("cpp", "grammars/cpp.json", "input/cpp.txt"),
    ScannerlessConfig("css", "grammars/css.json", "input/css.txt", max_source_bytes=850),
    ScannerlessConfig("html", "grammars/html.json", "input/html.txt", max_source_bytes=967),
    ScannerlessConfig("json", "grammars/json.json", "input/json.txt"),
    ScannerlessConfig("java", "grammars/jsl18.json", "input/java.txt"),
    ScannerlessConfig("pascal", "grammars/pascal.json", "input/pascal.txt"),
    ScannerlessConfig("tinypascal", "grammars/ll1_tinypascal.json", "input/tinypascal.txt"),
    ScannerlessConfig("sexp", "grammars/sexp.json", "input/sexp.txt"),
    ScannerlessConfig("shell", "grammars/shell.json", "input/shell.txt", max_source_bytes=1500),
    ScannerlessConfig("sql", "grammars/sql.json", "input/sql.txt"),
    ScannerlessConfig("tinyc", "grammars/tinyc.json", "input/tinyc.txt"),
    ScannerlessConfig("tinyc_lr", "grammars/lr_tinyc.json", "input/tinyc_lr.txt"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def derive_rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def load_valid_lines(root: Path, config: ScannerlessConfig) -> list[str]:
    lines: list[str] = []
    path = root / config.input_path
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            lines.append(line)
    if not lines:
        raise RuntimeError(f"{config.name}: no non-empty source lines found")
    return lines


def build_main_binary(root: Path) -> Path:
    subprocess.run(
        ["cargo", "build", "--release", "--quiet", "--bin", "parser_comparison"],
        cwd=root,
        check=True,
    )
    suffix = ".exe" if sys.platform.startswith("win") else ""
    binary = root / "target" / "release" / f"parser_comparison{suffix}"
    if not binary.exists():
        raise RuntimeError(f"expected main binary not found: {binary}")
    return binary


def accepted_by_main(binary: Path, root: Path, grammar_path: str, candidate: str) -> bool | None:
    try:
        completed = subprocess.run(
            [str(binary), "--text", candidate, grammar_path, "--char"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=VALIDATION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None

    if "Tokenization failed" in completed.stderr:
        return None
    if "Parse succeeded" in completed.stdout:
        return True
    if "Parse failed" in completed.stdout and "Tokens:" in completed.stdout:
        return False
    return None


def length_bounds(max_length: int, strata_count: int = STRATA_COUNT) -> tuple[tuple[int, int], ...]:
    if max_length < 1:
        raise ValueError("max_length must be positive")

    effective_strata = min(strata_count, max_length)
    bounds = []
    start = 1
    for index in range(effective_strata):
        end = ((index + 1) * max_length) // effective_strata
        end = max(start, end)
        if index == effective_strata - 1:
            end = max_length
        bounds.append((start, end))
        start = end + 1
    return tuple(bounds)


def quota_for(keys: tuple, total: int) -> dict:
    quota = {key: total // len(keys) for key in keys}
    for key in keys[: total % len(keys)]:
        quota[key] += 1
    return quota


def method_can_hit_bounds(lines: list[str], method: str, bounds: tuple[int, int], max_bytes: int) -> bool:
    lo, hi = bounds
    if method == "mutate_char":
        return any(lo <= len(line) <= hi and len(line.encode("utf-8")) <= max_bytes for line in lines)
    if method in ("cut_prefix", "cut_suffix"):
        return any(len(line) > lo for line in lines)
    raise ValueError(f"unknown generation method: {method}")


def plan_cell_quotas(
    lines: list[str],
    target_count: int,
    max_bytes: int,
    bounds_by_stratum: tuple[tuple[int, int], ...],
) -> dict[tuple[str, int], int]:
    method_quota = quota_for(METHODS, target_count)
    strata = tuple(range(len(bounds_by_stratum)))
    stratum_quota = quota_for(strata, target_count)
    cell_quota = {(method, stratum): 0 for method in METHODS for stratum in strata}

    remaining_strata = dict(stratum_quota)
    methods_by_scarcity = sorted(
        METHODS,
        key=lambda method: sum(
            method_can_hit_bounds(lines, method, bounds_by_stratum[stratum], max_bytes)
            for stratum in strata
        ),
    )

    for method in methods_by_scarcity:
        feasible = [
            stratum
            for stratum in strata
            if method_can_hit_bounds(lines, method, bounds_by_stratum[stratum], max_bytes)
        ]
        if not feasible:
            raise RuntimeError(f"{method} cannot generate candidates in any length stratum")
        for _ in range(method_quota[method]):
            available = [stratum for stratum in feasible if remaining_strata[stratum] > 0]
            if not available:
                raise RuntimeError(f"cannot satisfy length-stratified quota for {method}")
            stratum = max(available, key=lambda index: (remaining_strata[index], -index))
            cell_quota[(method, stratum)] += 1
            remaining_strata[stratum] -= 1

    if any(count != 0 for count in remaining_strata.values()):
        raise RuntimeError(f"unsatisfied length quotas: {remaining_strata}")
    return cell_quota


def choose_generation_goal(
    rng: random.Random,
    lines: list[str],
    method_counts: dict[str, int],
    method_quota: dict[str, int],
    stratum_counts: dict[int, int],
    stratum_quota: dict[int, int],
    bounds_by_stratum: tuple[tuple[int, int], ...],
    max_bytes: int,
) -> tuple[str, int, tuple[int, int]]:
    goals = []
    for method in METHODS:
        if method_counts[method] >= method_quota[method]:
            continue
        for stratum, bounds in enumerate(bounds_by_stratum):
            if stratum_counts[stratum] >= stratum_quota[stratum]:
                continue
            if method_can_hit_bounds(lines, method, bounds, max_bytes):
                goals.append((method, stratum, bounds))
    if not goals:
        raise RuntimeError("no feasible underfilled generation goal remains")
    return rng.choice(goals)


def choose_source_line(
    rng: random.Random,
    lines: list[str],
    method: str,
    max_bytes: int,
    bounds: tuple[int, int],
) -> str | None:
    lo, hi = bounds
    if method == "mutate_char":
        candidates = [
            line
            for line in lines
            if lo <= len(line) <= hi and len(line.encode("utf-8")) <= max_bytes
        ]
    else:
        candidates = [line for line in lines if len(line) > lo]
    if not candidates:
        return None
    return rng.choice(candidates)


def make_candidate(
    rng: random.Random,
    line: str,
    method: str,
    alphabet: tuple[str, ...],
    bounds: tuple[int, int],
) -> str | None:
    lo, hi = bounds
    if method in ("cut_prefix", "cut_suffix"):
        if len(line) <= lo:
            return None
        candidate_length = rng.randrange(lo, min(hi, len(line) - 1) + 1)
        if method == "cut_prefix":
            return line[:candidate_length]
        return line[len(line) - candidate_length :]

    if method == "mutate_char":
        if not line or not (lo <= len(line) <= hi) or len(alphabet) < 2:
            return None
        index = rng.randrange(len(line))
        original = line[index]
        replacements = [char for char in alphabet if char != original]
        if not replacements:
            return None
        replacement = rng.choice(replacements)
        return f"{line[:index]}{replacement}{line[index + 1:]}"

    raise ValueError(f"unknown generation method: {method}")


def keep_candidate(candidate: str, valid_set: set[str], seen: set[str], max_bytes: int) -> bool:
    if not candidate or not candidate.strip():
        return False
    if "\n" in candidate or "\r" in candidate:
        return False
    if len(candidate.encode("utf-8")) > max_bytes:
        return False
    if candidate in valid_set or candidate in seen:
        return False
    return True


def generate_for_config(
    root: Path,
    binary: Path,
    config: ScannerlessConfig,
    seed: int,
    target_count: int,
    max_bytes: int,
) -> dict[str, int]:
    # Apply the per-grammar source-length cap (see ScannerlessConfig.max_source_bytes).
    effective_max_bytes = min(max_bytes, config.max_source_bytes)
    lines = load_valid_lines(root, config)
    valid_set = set(lines)
    alphabet = tuple(sorted({char for line in lines for char in line}))
    rng = derive_rng(seed, config.name)
    max_source_length = min(max(len(line) for line in lines), effective_max_bytes)
    bounds_by_stratum = length_bounds(max_source_length)
    cell_quota = plan_cell_quotas(lines, target_count, effective_max_bytes, bounds_by_stratum)

    generated: list[str] = []
    seen: set[str] = set()
    cell_counts = {cell: 0 for cell in cell_quota}
    attempts = 0
    max_attempts = target_count * 200

    while len(generated) < target_count and attempts < max_attempts:
        attempts += 1
        underfilled_cells = [cell for cell, quota in cell_quota.items() if cell_counts[cell] < quota]
        method, stratum = rng.choice(underfilled_cells)
        bounds = bounds_by_stratum[stratum]
        line = choose_source_line(rng, lines, method, effective_max_bytes, bounds)
        if line is None:
            continue
        candidate = make_candidate(rng, line, method, alphabet, bounds)
        if candidate is None or not keep_candidate(candidate, valid_set, seen, effective_max_bytes):
            continue

        accepted = accepted_by_main(binary, root, config.grammar_path, candidate)
        if accepted is False:
            generated.append(candidate)
            seen.add(candidate)
            cell_counts[(method, stratum)] += 1

    if len(generated) != target_count:
        raise RuntimeError(
            f"{config.name}: generated {len(generated)} of {target_count} invalid inputs "
            f"after {attempts} attempts"
        )

    output_path = root / config.output_path
    output_path.write_text("\n".join(generated) + "\n", encoding="utf-8")
    method_counts = {method: 0 for method in METHODS}
    stratum_counts = {stratum: 0 for stratum in range(len(bounds_by_stratum))}
    for (method, stratum), count in cell_counts.items():
        method_counts[method] += count
        stratum_counts[stratum] += count
    return {
        **method_counts,
        **{f"stratum_{stratum + 1}": count for stratum, count in stratum_counts.items()},
        "attempts": attempts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--skip-build", action="store_true", help="reuse target/release/parser_comparison without rebuilding")
    parser.add_argument(
        "--config",
        action="append",
        choices=[config.name for config in SCANNERLESS_CONFIGS],
        help="limit generation to one or more grammar names",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    binary = root / "target" / "release" / ("parser_comparison.exe" if sys.platform.startswith("win") else "parser_comparison")
    if not args.skip_build:
        binary = build_main_binary(root)
    elif not binary.exists():
        raise RuntimeError(f"--skip-build requested but binary does not exist: {binary}")

    selected = [
        config
        for config in SCANNERLESS_CONFIGS
        if args.config is None or config.name in set(args.config)
    ]

    print(
        f"Generating {args.target_count} invalid inputs per grammar "
        f"(seed={args.seed}, max_bytes={args.max_bytes})"
    )
    for config in selected:
        counts = generate_for_config(
            root=root,
            binary=binary,
            config=config,
            seed=args.seed,
            target_count=args.target_count,
            max_bytes=args.max_bytes,
        )
        method_counts = ", ".join(f"{method}={counts[method]}" for method in METHODS)
        stratum_total = sum(1 for key in counts if key.startswith("stratum_"))
        stratum_counts = ", ".join(
            f"s{index + 1}={counts[f'stratum_{index + 1}']}" for index in range(stratum_total)
        )
        print(
            f"{config.name:12} -> {config.output_path} "
            f"({method_counts}; {stratum_counts}; attempts={counts['attempts']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
