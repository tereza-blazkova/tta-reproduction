"""Loaders for features, labels, metadata and per-session transcripts.

Two splits, three evaluation populations:

* **train** — 35,072 responses / 22,821 sessions, 100% Third Space Learning. Model fitting
  and the LO-disjoint out-of-fold view.
* **test TSL** — 9,640 responses / 7,373 sessions, same provider, released after the freeze.
* **test Eedi** — 868 responses / 868 sessions, typed chat, a provider absent from training,
  so zero-shot transfer rather than a subgroup.

`data_source` lives in `qualitative_metadata.csv`, not in the feature files; :func:`load_test`
merges it in, so no stage has to remember that.
"""
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from . import config

TRANSCRIPT_COLS = ["session_id", "utterance_id", "role", "content", "timestamp"]


def load_train() -> pd.DataFrame:
    """response_id, session_id, learning_objective_id, learning_objective, is_correct."""
    f = pd.read_csv(config.TRAIN_FEATURES, dtype=str)
    y = pd.read_csv(config.TRAIN_LABELS)
    y["is_correct"] = y["is_correct"].astype(float).astype(int)
    return f.merge(y, on="response_id", how="inner")


def load_test(with_labels: bool = True) -> pd.DataFrame:
    """The released test split, with `data_source` ∈ {tsl, eedi} merged in."""
    f = pd.read_csv(config.TEST_FEATURES, dtype=str)
    meta = pd.read_csv(config.METADATA, dtype=str)
    out = f.merge(meta[["response_id", "data_source", "student_id", "question_id"]],
                  on="response_id", how="left")
    if out["data_source"].isna().any():
        raise RuntimeError("test rows without a data_source — check qualitative_metadata.csv")
    if with_labels:
        y = pd.read_csv(config.TEST_LABELS)
        y["is_correct"] = y["is_correct"].astype(float).astype(int)
        out = out.merge(y, on="response_id", how="inner")
    return out


def load_metadata() -> pd.DataFrame:
    """response_id, student_id, tutor_id, question_id, data_source, split — both splits.

    `student_id` is what makes the within-student designs possible; on Eedi `tutor_id` is
    100% null, so any tutor-level analysis is TSL-only.
    """
    return pd.read_csv(config.METADATA, dtype=str)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSCRIPT_COLS)


def _parse(raw: bytes | str) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(raw) if isinstance(raw, bytes) else raw, dtype=str)
    df["utterance_id"] = pd.to_numeric(df["utterance_id"], errors="coerce")
    return df.sort_values("utterance_id").reset_index(drop=True)


class TranscriptStore:
    """Per-session transcripts from a zip of `{session_id}.csv` members, or a directory.

    `role` ∈ {tutor, student, background}: `background` is real (noise, `[unclear]`), and
    :func:`ttapub.dialogue.build_turns` is what drops it. `timestamp` is an elapsed offset
    `HH:MM:SS` from session start, not a wall clock.
    """

    def __init__(self, source: str | Path):
        self.source = Path(source)
        if self.source.is_dir():
            self.is_zip = False
            self._index = {p.stem: p for p in self.source.glob("*.csv")}
        elif self.source.is_file() and zipfile.is_zipfile(self.source):
            self.is_zip = True
            with zipfile.ZipFile(self.source) as zf:
                names = [n for n in zf.namelist() if n.endswith(".csv")]
            self._index = {n.rsplit("/", 1)[-1][:-4]: n for n in names}
        else:
            raise FileNotFoundError(
                f"not a transcript source: {self.source} (want a .zip or a directory)")

    def sessions(self) -> set[str]:
        return set(self._index)

    def __contains__(self, sid: str) -> bool:
        return sid in self._index

    def load(self, sid: str) -> pd.DataFrame:
        ref = self._index.get(sid)
        if ref is None:
            return _empty()
        if self.is_zip:
            with zipfile.ZipFile(self.source) as zf:
                return _parse(zf.read(ref))
        return _parse(str(ref))

    def iter(self, session_ids):
        """Yield (session_id, transcript) for the given ids, opening a zip once."""
        wanted = list(dict.fromkeys(session_ids))
        if not self.is_zip:
            for sid in wanted:
                yield sid, self.load(sid)
            return
        with zipfile.ZipFile(self.source) as zf:
            for sid in wanted:
                ref = self._index.get(sid)
                yield sid, (_parse(zf.read(ref)) if ref else _empty())


def train_store() -> TranscriptStore:
    return TranscriptStore(config.TRAIN_TRANSCRIPTS)


def test_store() -> TranscriptStore:
    return TranscriptStore(config.TEST_TRANSCRIPTS)


def store(split: str) -> TranscriptStore:
    return train_store() if split == "train" else test_store()
