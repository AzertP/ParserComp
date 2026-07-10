#!/usr/bin/env python3
"""Generate reproducible synthetic scannerless benchmark inputs."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.fuzzer import LimitFuzzer


DEFAULT_SEED = 20260709
DEFAULT_TARGET_COUNT = 150
DEFAULT_CANDIDATE_COUNT = 1000
DEFAULT_MAX_ATTEMPTS = 50000
DEFAULT_STRATA = 5
DEFAULT_MIN_DEPTH = 1
DEFAULT_MAX_DEPTH = 30
DEFAULT_SIMPLE_LIMIT = 50000
DEFAULT_COMPLEX_LIMIT = 10000

SIMPLE_BUCKET = "simple"
COMPLEX_BUCKET = "complex"


@dataclass(frozen=True)
class SyntheticConfig:
    name: str
    source_path: str
    generator_stem: str
    bucket: str
    max_depth: int | None = None
    min_depth: int | None = None

    @property
    def copied_grammar_path(self) -> str:
        return f"generator/grammars/{self.generator_stem}.json"

    @property
    def output_path(self) -> str:
        return f"generator/input/{self.name}.txt"


TARGET_CONFIGS: tuple[SyntheticConfig, ...] = (
    SyntheticConfig(
        "sexp",
        "grammars/sexp.json",
        "sexp",
        SIMPLE_BUCKET,
        max_depth=30,
    ),
    SyntheticConfig("expr", "grammars/calc.json", "expr", SIMPLE_BUCKET, max_depth=24),
    SyntheticConfig(
        "json",
        "grammars/json.json",
        "json",
        SIMPLE_BUCKET,
        max_depth=30,
    ),
    SyntheticConfig(
        "tinyc",
        "grammars/tinyc.json",
        "tinyc",
        SIMPLE_BUCKET,
        max_depth=26,
    ),
    SyntheticConfig(
        "bool",
        "grammars/bool.json",
        "bool",
        SIMPLE_BUCKET,
        max_depth=22,
    ),
    SyntheticConfig("ansi_c", "grammars/ansi_c.json", "ansi_c", COMPLEX_BUCKET, max_depth=18),
    SyntheticConfig("cpp", "grammars/cpp.json", "cpp", COMPLEX_BUCKET, max_depth=18),
    SyntheticConfig("pascal", "grammars/pascal.json", "pascal", COMPLEX_BUCKET, max_depth=26),
    SyntheticConfig("java", "grammars/jsl18.json", "java", COMPLEX_BUCKET, max_depth=26),
    SyntheticConfig("html", "grammars/html.json", "html", COMPLEX_BUCKET, max_depth=30),
    SyntheticConfig("css", "grammars/css.json", "css", COMPLEX_BUCKET, max_depth=26),
    SyntheticConfig("shell", "grammars/shell.json", "shell", COMPLEX_BUCKET, max_depth=18),
    SyntheticConfig("sql", "grammars/sql.json", "sql", COMPLEX_BUCKET, max_depth=14),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def derive_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def length_limit_for(config: SyntheticConfig, simple_limit: int, complex_limit: int) -> int:
    if config.bucket == SIMPLE_BUCKET:
        return simple_limit
    if config.bucket == COMPLEX_BUCKET:
        return complex_limit
    raise ValueError(f"{config.name}: unknown bucket {config.bucket}")


def depth_range_for(config: SyntheticConfig, args: Namespace) -> tuple[int, int]:
    min_depth = args.min_depth
    if config.min_depth is not None:
        min_depth = max(min_depth, config.min_depth)
    max_depth = args.max_depth
    if config.max_depth is not None:
        max_depth = min(max_depth, config.max_depth)
    if max_depth < min_depth:
        raise ValueError(
            f"{config.name}: max depth {max_depth} is below min depth {min_depth}"
        )
    return min_depth, max_depth


def copy_grammars(
    root: Path,
    configs: Sequence[SyntheticConfig],
    overwrite: bool = False,
) -> None:
    grammar_dir = root / "generator" / "grammars"
    grammar_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        source = root / config.source_path
        destination = root / config.copied_grammar_path
        if not source.exists():
            raise FileNotFoundError(f"{config.name}: missing source grammar {source}")
        if destination.exists() and not overwrite:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open(encoding="utf-8") as handle:
            grammar = json.load(handle)
        destination.write_text(json.dumps(grammar, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_acceptable_candidate(candidate: str, seen: set[str], max_length: int) -> bool:
    return (
        bool(candidate)
        and candidate not in seen
        and "\n" not in candidate
        and "\r" not in candidate
        and len(candidate) < max_length
    )


def _length_bounds(candidates: Sequence[str], strata_count: int) -> list[tuple[int, int]]:
    lengths = [len(candidate) for candidate in candidates]
    min_length = min(lengths)
    max_length = max(lengths)
    span = max_length - min_length + 1
    base = span // strata_count
    remainder = span % strata_count
    bounds = []
    start = min_length
    for index in range(strata_count):
        width = base + (1 if index < remainder else 0)
        end = start + max(width, 1) - 1
        bounds.append((start, end))
        start = end + 1
    return bounds


def spread_sample(candidates: Sequence[str], target_count: int) -> list[str]:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    unique_candidates = sorted(set(candidates), key=lambda item: (len(item), item))
    if len(unique_candidates) < target_count:
        raise ValueError(
            f"need at least {target_count} candidates, found {len(unique_candidates)}"
        )

    if target_count == 1:
        return [unique_candidates[len(unique_candidates) // 2]]

    last_index = len(unique_candidates) - 1
    denominator = target_count - 1
    selected = [
        unique_candidates[(index * last_index) // denominator]
        for index in range(target_count)
    ]
    return sorted(selected, key=lambda item: (len(item), item))


def sample_for_config(
    candidates: Sequence[str],
    _config: SyntheticConfig,
    args: Namespace,
    _rng: random.Random,
) -> list[str]:
    return spread_sample(candidates, args.target_count)


def median_length(lines: Sequence[str]) -> float:
    lengths = sorted(len(line) for line in lines)
    midpoint = len(lengths) // 2
    if len(lengths) % 2:
        return float(lengths[midpoint])
    return (lengths[midpoint - 1] + lengths[midpoint]) / 2.0


def stratum_counts(
    lines: Sequence[str],
    strata_count: int,
    bounds: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, int]]:
    if bounds is None:
        bounds = _length_bounds(lines, strata_count)
    counts = []
    for lo, hi in bounds:
        count = sum(1 for line in lines if lo <= len(line) <= hi)
        counts.append({"min": lo, "max": hi, "count": count})
    return counts


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_lines(path: Path, lines: Sequence[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return sha256_text(content)


def generate_for_config(root: Path, config: SyntheticConfig, args: Namespace) -> dict:
    grammar_path = root / config.copied_grammar_path
    grammar_json = _load_json(grammar_path)
    rules = grammar_json["rules"]
    start = grammar_json["start"]
    rule_weights = grammar_json.get("rule-weights")
    max_length = length_limit_for(config, args.simple_limit, args.complex_limit)
    rng = random.Random(derive_seed(args.seed, config.name))
    fuzzer = LimitFuzzer(
        rules,
        rule_weights=rule_weights,
        rng=rng,
        bias_long=True,
    )

    seen: set[str] = set()
    candidates: list[str] = []
    min_depth, max_depth = depth_range_for(config, args)
    depth_span = max_depth - min_depth + 1

    attempts = 0
    while attempts < args.max_attempts:
        if len(candidates) >= args.candidate_count:
            break
        depth = min_depth + (attempts % depth_span)
        attempts += 1
        try:
            candidate = fuzzer.iter_fuzz(start, max_depth=depth, max_length=max_length)
        except (RuntimeError, RecursionError):
            continue
        if is_acceptable_candidate(candidate, seen, max_length):
            seen.add(candidate)
            candidates.append(candidate)

    if len(candidates) < args.target_count:
        raise RuntimeError(
            f"{config.name}: generated {len(candidates)} acceptable candidates "
            f"after {attempts} attempts; need {args.target_count}"
        )

    selected = sample_for_config(
        candidates,
        config,
        args,
        random.Random(derive_seed(args.seed, f"{config.name}:sample")),
    )
    output_path = root / config.output_path
    output_hash = _write_lines(output_path, selected)
    grammar_hash = sha256_text(grammar_path.read_text(encoding="utf-8"))
    lengths = [len(line) for line in selected]

    return {
        "name": config.name,
        "bucket": config.bucket,
        "source_grammar": config.source_path,
        "generator_grammar": config.copied_grammar_path,
        "generator_grammar_sha256": grammar_hash,
        "output_path": config.output_path,
        "output_sha256": output_hash,
        "length_limit": max_length,
        "attempts": attempts,
        "unique_accepted_candidates": len(candidates),
        "selected_count": len(selected),
        "min_length": min(lengths),
        "median_length": median_length(selected),
        "max_length": max(lengths),
        "strata": stratum_counts(selected, args.strata),
        "strata_mode": "candidate-spread",
        "min_depth": min_depth,
        "max_depth": max_depth,
        "rule_weighted_symbols": sorted((rule_weights or {}).keys()),
    }


def selected_configs(grammar_names: Sequence[str] | None) -> list[SyntheticConfig]:
    if not grammar_names:
        return list(TARGET_CONFIGS)
    by_name = {config.name: config for config in TARGET_CONFIGS}
    missing = sorted(set(grammar_names) - set(by_name))
    if missing:
        raise ValueError(f"unknown grammar name(s): {', '.join(missing)}")
    return [by_name[name] for name in grammar_names]


def manifest_path(root: Path) -> Path:
    return root / "generator" / "input" / "manifest.json"


def write_manifest(root: Path, args: Namespace, entries: Sequence[dict]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "target_count": args.target_count,
        "candidate_count": args.candidate_count,
        "max_attempts": args.max_attempts,
        "strata": args.strata,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "simple_limit": args.simple_limit,
        "complex_limit": args.complex_limit,
        "grammars": list(entries),
    }
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(root: Path) -> dict:
    path = manifest_path(root)
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    return _load_json(path)


def verify_outputs(root: Path, manifest: dict) -> None:
    for entry in manifest["grammars"]:
        output_path = root / entry["output_path"]
        if not output_path.exists():
            raise FileNotFoundError(f"{entry['name']}: missing output {output_path}")
        content = output_path.read_text(encoding="utf-8")
        if sha256_text(content) != entry["output_sha256"]:
            raise RuntimeError(f"{entry['name']}: output hash mismatch")
        lines = content.splitlines()
        if len(lines) != entry["selected_count"]:
            raise RuntimeError(
                f"{entry['name']}: expected {entry['selected_count']} lines, found {len(lines)}"
            )
        if any(not line for line in lines):
            raise RuntimeError(f"{entry['name']}: empty output line")
        if len(set(lines)) != len(lines):
            raise RuntimeError(f"{entry['name']}: duplicate output lines")
        if max(len(line) for line in lines) >= entry["length_limit"]:
            raise RuntimeError(f"{entry['name']}: output exceeds length limit")


def print_summary(manifest: dict) -> None:
    for entry in manifest["grammars"]:
        print(
            f"{entry['name']}: count={entry['selected_count']} "
            f"lengths={entry['min_length']}-{entry['max_length']} "
            f"median={entry['median_length']:.1f} "
            f"candidates={entry['unique_accepted_candidates']} "
            f"attempts={entry['attempts']}"
        )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--strata", type=int, default=DEFAULT_STRATA)
    parser.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--simple-limit", type=int, default=DEFAULT_SIMPLE_LIMIT)
    parser.add_argument("--complex-limit", type=int, default=DEFAULT_COMPLEX_LIMIT)
    parser.add_argument("--grammar", action="append", dest="grammars")
    parser.add_argument("--refresh-grammars", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()

    if args.verify_only:
        manifest = read_manifest(root)
        verify_outputs(root, manifest)
        print(f"Verified {len(manifest['grammars'])} generated corpora.")
        return 0

    if args.summary:
        print_summary(read_manifest(root))
        return 0

    configs = selected_configs(args.grammars)
    copy_grammars(root, configs, overwrite=args.refresh_grammars)
    entries = []
    for config in configs:
        print(f"{config.name}: generating...", flush=True)
        entry = generate_for_config(root, config, args)
        entries.append(entry)
        print(
            f"{config.name}: selected {entry['selected_count']} "
            f"from {entry['unique_accepted_candidates']} candidates "
            f"(lengths {entry['min_length']}-{entry['max_length']})",
            flush=True,
        )
    write_manifest(root, args, entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
