"""Tabular foundation models behind scikit-learn's ``fit`` / ``predict_proba``.

In-context learners, pretrained only on synthetic tables, none fine-tuned here:

* **TabPFN 3.5** (`Prior-Labs/tabpfn_3_5`, ``tabpfn-v3.5-20260909.safetensors``), the library's
  default, as ``tabpfn35``. Public on Hugging Face (rev ``06bf2ba``) under the TabPFN-3.5
  Non-Commercial License, which permits academic research and benchmarking but no commercial use
  of the model or its outputs: fine for the paper's comparison, not for anything shipped. Its row
  limit lives in the checkpoint (1,000,000), so the context here is inside it.

**What "fit" means.** No weight is updated. ``fit`` stores the training rows as the context;
``predict_proba`` attends from every query row to that context. So a fold's model is its
training rows, and the test model is all 35,072 training rows.

**Seeds.** Each library's ensemble (`n_est` members: column shuffles, transforms, class
shifts) is itself random. One run moves clean-row log loss by ~0.001 between seeds,
the size of a small block effect, so the probability is the mean over ``seeds`` runs, as the
XGBoost recipe averages its seeds.

**Query rows must not see each other** (the paper's no-pooling rule:
a test row's prediction may depend on the training rows and on itself only). The
architectures promise that query rows attend to context rows only, but promises about
preprocessing are where pooling leaks in. :func:`independence_gate` measures it instead.

Weights are read from the local Hugging Face cache (``HF_HOME``); set ``HF_HUB_OFFLINE=1`` on a
node without internet. They need a CUDA GPU.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

LEARNERS = ("tabpfn35",)
TABPFN35_REPO = "Prior-Labs/tabpfn_3_5"
TABPFN35_CKPT = "tabpfn-v3.5-20260909.safetensors"


def _p1(proba: np.ndarray, classes) -> np.ndarray:
    """The column of P(correct), found by class label rather than assumed to be the last."""
    classes = [int(float(c)) for c in classes]
    if sorted(classes) != [0, 1]:
        raise SystemExit(f"expected classes {{0, 1}}, the model returned {classes}")
    p = np.asarray(proba, dtype=float)[:, classes.index(1)]
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise SystemExit("the model returned values that are not probabilities")
    return p


class _InContext(ClassifierMixin, BaseEstimator):
    """Shared plumbing: store the context, average the seeds, report what was run.

    A scikit-learn estimator (sklearn >= 1.6 refuses a pipeline step without its tags), so every
    ``__init__`` argument is stored under its own name for ``get_params``."""

    name = "?"

    def __init__(self, n_est: int = 8, seeds=(42,), device: str = "cuda", fp32: bool = False):
        self.n_est, self.seeds, self.device, self.fp32 = n_est, list(seeds), device, fp32

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xq = np.asarray(X, dtype=np.float32)
        # each seed's probabilities are kept, so a block comparison can carry the model's own
        # run-to-run randomness in its interval (27_paired_drop.py --seed-spread)
        self.seed_p_ = {s: self._predict(Xq, s) for s in self.seeds}
        p = np.mean(list(self.seed_p_.values()), axis=0)
        return np.column_stack([1 - p, p])

    def _predict(self, Xq: np.ndarray, seed: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:
        return {"learner": self.name, "n_est": self.n_est, "seeds": self.seeds,
                "precision": "fp32" if self.fp32 else "library default"}




class TabPFN35(_InContext):
    """TabPFN 3.5 through its scikit-learn estimator, library defaults except the seed and the
    checkpoint.

    ``ignore_pretraining_limits`` stays on so a context past the checkpoint's row limit runs
    instead of stopping; ``row_limit_`` records that limit so the table can say which side of
    it the 28k-35k rows fall."""

    name = "tabpfn35"

    @staticmethod
    def _weights() -> str:
        """The checkpoint as a real ``.safetensors`` file. tabpfn resolves symlinks before it
        reads the suffix, and a Hugging Face cache blob has none, so the cached path would be
        unpickled as a torch file. A link beside the blob, named by its own hash, fixes that."""
        import os
        import shutil
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        blob = Path(hf_hub_download(TABPFN35_REPO, TABPFN35_CKPT, local_files_only=True)).resolve()
        path = blob.with_name(blob.name + ".safetensors")
        if not path.exists():
            try:
                os.link(blob, path)
            except OSError:
                shutil.copyfile(blob, path)
        return str(path)

    def _predict(self, Xq: np.ndarray, seed: int) -> np.ndarray:
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, model_path=self._weights(), n_estimators=self.n_est, device=self.device,
            ignore_pretraining_limits=True, random_state=seed,
            inference_precision=torch.float32 if self.fp32 else "auto")
        clf.fit(self.X_, self.y_)
        if not hasattr(self, "row_limit_"):
            self.row_limit_ = int(clf.inference_config_.MAX_NUMBER_OF_SAMPLES)
            print(f"  tabpfn35: context {len(self.X_):,} rows, checkpoint row limit "
                  f"{self.row_limit_:,}", flush=True)
        return _p1(clf.predict_proba(Xq), clf.classes_)

    def describe(self) -> dict:
        return super().describe() | {"weights": f"{TABPFN35_REPO}:{TABPFN35_CKPT}",
                                     "licence": "TabPFN-3.5 Non-Commercial License"}


def make(kind: str, n_est: int, seeds, size: str = "small", fp32: bool | None = None):
    """A learner by name. ``fp32=None`` keeps each learner's default precision."""
    cls = {"tabpfn35": TabPFN35}[kind]
    kw = {} if fp32 is None else {"fp32": fp32}
    return cls(n_est=n_est, seeds=seeds, **kw)


def independence_gate(factory, Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray,
                      n_ctx: int = 4096, n_head: int = 256, seed: int = 0) -> dict:
    """Do query rows influence each other's predictions?

    The first ``n_head`` query rows are predicted three ways: alone, beside the next 768 query
    rows, and beside a different 768. Each call rebuilds every input from the rows it is given,
    so per-call preprocessing over the query set is caught along with attention between query
    rows. A context of ``n_ctx`` training rows keeps it cheap; independence does not depend on
    the context size. Returns the largest disagreement for fp32 and for the learner's
    production precision: fp32 must agree to float noise, and the production number shows what
    reduced precision adds (batch-shape kernel noise, not dependence).
    """
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(Xc), size=min(n_ctx, len(Xc)), replace=False)
    head = Xq[:n_head]
    side_a, side_b = Xq[n_head:n_head + 768], Xq[n_head + 768:n_head + 1536]
    out = {}
    for label, fp32 in (("fp32", True), ("production", None)):
        m = factory(fp32).fit(Xc[ix], yc[ix])
        alone = m.predict_proba(head)[:, 1]
        a = m.predict_proba(np.vstack([head, side_a]))[:n_head, 1]
        b = m.predict_proba(np.vstack([head, side_b]))[:n_head, 1]
        out[label] = float(max(np.abs(alone - a).max(), np.abs(alone - b).max()))
    return out
