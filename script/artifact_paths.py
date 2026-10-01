"""Shared input and output paths for the artifact generators."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results"
COMBINED_DIR = PROJECT_ROOT / "results_comprehensive"
GRAMMARS_DIR = PROJECT_ROOT / "grammars"
IMG_DIR = RESULT_ROOT / "tex"
