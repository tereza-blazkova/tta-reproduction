"""Per-turn labels (or probabilities) -> one session rate per code, the only aggregation here.

**What a turn carries, and it differs by path.** On the **main path** (stage 05
``--source labels``) the annotator assigns each turn **exactly one** code, so the per-turn value
is a 0/1 indicator and a session rate is literally the share of that role's turns the LLM gave
the code. On the **appendix path** (``--source probs``) the distilled children emit a
probability per code per turn, and the rate is their mean. Everything below is written for the
general case because the arithmetic is identical --- the indicator is just the degenerate
probability --- but when reading a number out of this module on the main path, P is 0 or 1 and
"rate" means "share of turns", with no soft weighting anywhere.

The annotation also records an optional second code per turn in its ``also`` field. **It never
reaches a feature**: stage 05 reads ``code`` only. ``also`` exists to measure how often two
codes collide, which is a property of the codebook rather than of a session.

**A rate, never a count.** A session's column for code *c* is the mean of the per-turn values of
*c* over that session's turns of the coded role — the share of turns showing the code. Rates transfer across registers; counts do not, and the test split
mixes 263-turn spoken sessions with 27-turn typed ones.

**Shrinkage, and why it is not tuning.** A rate from a short session is noisy, with sampling sd
falling as 1/sqrt(n), so the booster would learn split thresholds that mean different things at
different session lengths. Each rate is therefore shrunk toward the corpus prior for its code
with a pseudo-count of kappa = 25 turns:

    r' = (sum_t P(c | turn t) + kappa * prior_c) / (n_turns + kappa)

where on the main path sum_t P(c | turn t) is simply the count of that session's turns labelled
c, so the numerator is "times it happened, plus 25 turns' worth of the corpus average".

kappa = 25 is where sqrt(n)/(n + kappa) peaks over the length range the test set spans, i.e. the
value that flattens the rate's sampling sd across lengths; it is not swept on any evaluation
fold. Within this corpus the shrinkage is close to a per-code affine rescaling (which a tree
model is invariant to); its effect is concentrated on short sessions, which is where the failure
it fixes was measured.

**The prior is an artifact, not a constant.** `CODE_PRIOR` for v3 is computed once from the new
annotation's own per-turn probabilities over all scored turns (stage 05) and written to
`data/derived/code_prior_v3.json`. It must be regenerated whenever the children change, and the
run that produced it is recorded beside it — a prior carried over from a different annotator is
a silent mismatch between a session's own evidence and what it is shrunk toward.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

#: Pseudo-count in turns: how much evidence a session needs before its own rate outweighs the
#: corpus prior. Frozen; see the module docstring for why 25 and not a swept value.
SHRINK_KAPPA = 25.0

#: Column naming. `v3t_` = tutor move, `v3s_` = student signal; both are session rates.
PREFIX = {"tutor": "v3t_", "student": "v3s_"}


def column(code: str, role: str) -> str:
    return f"{PREFIX[role]}{code.lower()}_rate"


def code_columns(cols, role: str | None = None) -> list[str]:
    pres = tuple(PREFIX.values()) if role is None else (PREFIX[role],)
    return [c for c in cols if c.startswith(pres) and c.endswith("_rate")]


def compute_prior(turn_probs: pd.DataFrame, codes: list[str]) -> dict[str, float]:
    """The corpus mean of each code's per-turn value, over every scored turn.

    On the main path that value is a 0/1 indicator, so the prior is the share of the corpus's
    turns of that role carrying the code — the same number the paper's column table prints.
    """
    return {c: float(turn_probs[c].mean()) for c in codes}


def save_prior(prior: dict[str, float], provenance: dict, path: Path | None = None) -> Path:
    path = path or config.CODE_PRIOR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kappa": SHRINK_KAPPA, "provenance": provenance,
                                "prior": prior}, indent=1))
    return path


def load_prior(path: Path | None = None) -> dict[str, float]:
    path = path or config.CODE_PRIOR
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run stage 05 to build the v3 code prior.")
    return json.loads(path.read_text())["prior"]


def shrunk_rates(turn_probs: pd.DataFrame, codes: list[str], role: str,
                 prior: dict[str, float], kappa: float = SHRINK_KAPPA,
                 session_col: str = "session_id") -> pd.DataFrame:
    """Session rates for one role's codes, shrunk toward `prior`.

    `turn_probs` must already be restricted to the turns of the coded role (tutor codes on
    turns with tutor text, student codes on turns with student text): the denominator is that
    role's turn count, and mixing the axes silently changes what every rate means.
    """
    missing = [c for c in codes if c not in prior]
    if missing:
        raise KeyError(f"no prior for {missing} — regenerate the prior with these codes")
    g = turn_probs.groupby(session_col)[codes]
    tot, n = g.sum(), g.size()
    pri = np.array([prior[c] for c in codes])
    r = (tot.to_numpy() + kappa * pri) / (n.to_numpy()[:, None] + kappa)
    return pd.DataFrame(r, index=tot.index,
                        columns=[column(c, role) for c in codes])


def session_features(turn_probs: pd.DataFrame, tutor_codes: list[str],
                     student_codes: list[str], prior: dict[str, float],
                     kappa: float = SHRINK_KAPPA) -> pd.DataFrame:
    """The full conversation block: one shrunk rate per code, one row per session.

    Requires `has_teacher` / `has_student` flags on `turn_probs` (see
    :func:`ttapub.dialogue.turn_frame`), which define each role's turn axis.
    """
    t = shrunk_rates(turn_probs[turn_probs.has_teacher], tutor_codes, "tutor", prior, kappa)
    s = shrunk_rates(turn_probs[turn_probs.has_student], student_codes, "student", prior, kappa)
    out = t.join(s, how="outer")
    out.index.name = "session_id"
    return out.reset_index()


def unshrunk_rates(turn_probs: pd.DataFrame, codes: list[str], role: str,
                   session_col: str = "session_id") -> pd.DataFrame:
    """Raw session means, for the sensitivity arm that reports what shrinkage changed."""
    m = turn_probs.groupby(session_col)[codes].mean()
    m.columns = [column(c, role) for c in codes]
    return m
