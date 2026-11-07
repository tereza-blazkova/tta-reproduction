"""Paths. Every stage reads and writes through here, so a move is one edit.

Layout:
    data/raw/        the competition data, copied in with normalised names
    data/folds/      the frozen LO-disjoint fold vector and clean-row mask
    data/reference/  prior-run tables used only as comparison baselines
    data/derived/    anything this repo computes and reuses (turn tables, labels, features)
    results/<run>/   one directory per run: metrics.json + predictions + provenance.json
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TTAPUB_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"
FOLDS = DATA / "folds"
REFERENCE = DATA / "reference"
DERIVED = DATA / "derived"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ── raw competition files (normalised names; see data/README.md for provenance) ──
TRAIN_FEATURES = RAW / "train_features.csv"
TRAIN_LABELS = RAW / "train_labels.csv"
TRAIN_TRANSCRIPTS = RAW / "train_transcripts.zip"
TEST_FEATURES = RAW / "test_features.csv"
TEST_LABELS = RAW / "test_labels.csv"
TEST_TRANSCRIPTS = RAW / "test_transcripts.zip"
METADATA = RAW / "qualitative_metadata.csv"

# ── frozen partition ──
FOLD_VECTOR = FOLDS / "folds_gkf5_lo.npy"
CLEAN_MASK = FOLDS / "clean_rows_gkf5.npy"

# ── derived artifacts, by stage ──
TURNS_TRAIN = DERIVED / "turns_train.parquet"        # stage 01
TURNS_TEST = DERIVED / "turns_test.parquet"          # stage 01
LABELS_V3 = DERIVED / "labels_v3"                    # stage 02 (one CSV per shard/role)
TURN_PROBS_TRAIN = DERIVED / "turn_probs_v3_train.parquet"   # stage 04
TURN_PROBS_TEST = DERIVED / "turn_probs_v3_test.parquet"     # stage 04
CHILDREN = DERIVED / "children_v3.joblib"            # stage 04
#: Second-annotator robustness: TTAPUB_ANNOTATOR=<name> points every feature table and prior at
#: `features_v3_<name>_*.csv` / `code_prior_v3_<name>.json`, so stage 05 and every stage after it
#: run unchanged on another annotator's labels. Unset (the main path), the paths are the 122B's,
#: exactly as before — a second annotator can never overwrite them.
ANNOTATOR = os.environ.get("TTAPUB_ANNOTATOR", "").strip()
_SFX = f"_{ANNOTATOR}" if ANNOTATOR else ""
CODE_PRIOR = DERIVED / f"code_prior_v3{_SFX}.json"          # stage 05 (labels instrument)
FEATURES_TRAIN = DERIVED / f"features_v3{_SFX}_train.csv"   # stage 05 (labels instrument)
FEATURES_TEST = DERIVED / f"features_v3{_SFX}_test.csv"     # stage 05 (labels instrument)


#: The two measurement instruments write to separate files and carry separate priors. They must
#: never share a path: a feature table whose provenance is ambiguous ("was this the 122B's
#: labels or a classifier's guess at them?") cannot be reasoned about, and the whole point of
#: the comparison is that the two are different measurements of the same construct.
def features_path(split: str, source: str = "labels") -> Path:
    stem = "features_v3" if source == "labels" else "features_v3dst"
    return DERIVED / f"{stem}{_SFX}_{split}.csv"


def prior_path(source: str = "labels") -> Path:
    stem = "code_prior_v3" if source == "labels" else "code_prior_v3dst"
    return DERIVED / f"{stem}{_SFX}.json"



def ensure(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
