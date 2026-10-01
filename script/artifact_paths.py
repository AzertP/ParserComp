"""Shared paths for the package copies of the manuscript generators."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results"
COMBINED_DIR = PROJECT_ROOT / "results_comprehensive"
GRAMMARS_DIR = PROJECT_ROOT / "grammars"
IMG_DIR = PROJECT_ROOT / "manuscript" / "img"
