"""Guided-JSON helpers for the vLLM annotation pass, plus the guards that make a run shippable.

**Why guided decoding.** With the label vocabulary enforced at decode time, an off-codebook
code is unreachable rather than filtered afterwards, and a malformed object cannot be emitted.
Unguided decoding can emit off-codebook codes and malformed objects; none of that is possible
here. The guards below still run, because the structured-output path
can be switched off for a model whose tokenizer breaks the grammar backend (`SO_OFF=1`), and
because a truncated generation is a silent failure the schema cannot catch.

**The three failure modes this module refuses to hide.**

1. *Truncation* — :func:`extract_json` returns None rather than guessing at a cut array, so the
   chunk is counted as a parse failure instead of contributing partial labels.
2. *Vocabulary drift* — :class:`VocabAudit` counts every code the model emits that is not in
   the pool. A systematic drift means the prompt and the schema disagree.
3. *A degraded run reaching the distiller* — the caller exits non-zero when the parse-failure
   rate exceeds a threshold or the audit trips, so a half-labelled corpus cannot quietly become
   a feature table.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

#: Set SO_OFF=1 to drop structured outputs (needed for a tokenizer whose merges break the
#: grammar backend). The audit below is then the only thing standing between a drifting model
#: and the distiller, so never set it for a production pass without reading the audit report.
SO_OFF = os.environ.get("SO_OFF", "0") == "1"


def engine_kwargs() -> dict:
    """Extra `vllm.LLM(...)` kwargs. Kept as a function so a backend switch is one edit."""
    kw: dict = {}
    backend = os.environ.get("VLLM_SO_BACKEND")
    if backend and not SO_OFF:
        kw["structured_outputs_config"] = {"backend": backend}
    return kw


def make_sampling_params(schema: dict | None, max_tokens: int):
    """Greedy decode, optionally constrained to `schema`.

    Greedy (temperature 0) is deliberate: the annotation is a measurement instrument, and a
    sampled label would make the same turn get different codes on a re-run, so the per-code
    rates would carry decoder noise on top of model error.
    """
    from vllm import SamplingParams

    kw: dict = {"temperature": 0.0, "max_tokens": max_tokens}
    if schema is not None and not SO_OFF:
        try:                                  # vLLM >= 0.11
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs"] = StructuredOutputsParams(json=schema)
        except ImportError:                   # older guided-decoding API
            from vllm.sampling_params import GuidedDecodingParams
            kw["guided_decoding"] = GuidedDecodingParams(json=schema)
    return SamplingParams(**kw)


_ARRAY = re.compile(r"\[.*\]", re.S)


def extract_json(text: str):
    """The JSON array in a generation, or None if it is absent or truncated.

    Returning None on truncation rather than repairing the string is what makes a cut chunk
    *countable*. A repaired array looks like a successful chunk with fewer labels, which reads
    downstream as "the model found nothing there".
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if text.count("```") >= 2 else text[3:]
        text = text[4:] if text.startswith("json") else text
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _ARRAY.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class VocabAudit:
    """Counts codes the model emitted that are not in the pool."""

    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.bad: Counter = Counter()
        self.seen = 0

    def check(self, codes) -> list[str]:
        """The subset of `codes` that is in the pool; anything else is recorded."""
        out = []
        for c in codes:
            self.seen += 1
            if c in self.allowed:
                out.append(c)
            else:
                self.bad[str(c)] += 1
        return out

    @property
    def rate(self) -> float:
        return self.bad.total() / max(1, self.seen)

    def report(self, label: str = "", threshold: float = 0.002) -> int:
        """Print the audit; return 1 if off-vocabulary output exceeds `threshold`."""
        if not self.bad:
            print(f"vocab audit{label}: clean ({self.seen:,} codes)", flush=True)
            return 0
        top = ", ".join(f"{k}x{v}" for k, v in self.bad.most_common(8))
        print(f"vocab audit{label}: {self.bad.total():,}/{self.seen:,} off-vocabulary "
              f"({self.rate:.2%}) — {top}", flush=True)
        return int(self.rate > threshold)
