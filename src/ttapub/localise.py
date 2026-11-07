"""Locating the stretch of a session that handled ONE learning objective.

A session runs ~100 turns and often covers several objectives, but a response is graded on one
of them. Session-wide features therefore credit an exchange that happened while working on
objective A for the answer to objective B. Fixing that is the prerequisite
for the strongest design this data supports:
once features vary *within* a session, a **session fixed effect** becomes estimable, which
removes the student's state that day, the tutor's identity and the student's traits at once —
none of which a student fixed effect can touch.

**Boundaries are found in turn space** rather than raw-utterance space, because this repo's turn table is the join key for every annotation and carries no
utterance index. The two thresholds are converted accordingly — a turn is one tutor block plus
one student block, so utterance counts roughly halve — and :func:`segmentation_report` checks
the result against the tutor's own stated lesson count.

**Boundaries are discourse acts, never objective vocabulary** ("moving on to the next lesson",
"our second lesson is…"). That matters: a segmenter cued by the objective's own words would
assign a segment to an objective by construction, and the matching step below would then be
circular. Slide-title cues were tried in the original and rejected — they split one lesson into
its slides.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")

#: Content words of an objective, minus the vocabulary every objective shares.
STOPS = {"the", "and", "for", "with", "from", "into", "than", "that", "this", "these",
         "those", "using", "use", "used", "how", "what", "which", "when", "where", "who",
         "can", "will", "are", "was", "were", "been", "have", "has", "had", "not", "you",
         "your", "our", "its", "it's", "they", "their", "them", "one", "two", "three",
         "numbers", "number", "problems", "problem", "involving", "given", "make", "making",
         "understand", "understanding", "identify", "recognise", "recognize", "solve",
         "solving", "find", "finding", "work", "working", "out", "know", "knowing", "able",
         "apply", "applying", "including", "represent", "explain", "calculate", "value",
         "values", "between", "each", "all", "any", "more", "most", "some", "such"}

_MOVE = (r"(?:mov\w+|go\w*|jump\w*|start\w*|begin\w*|continu\w*|proceed\w*|carry\s+on|"
         r"head\w*|switch\w*)")
BOUND = re.compile(
    rf"{_MOVE}\s+(?:on\s+)?(?:to|with|into|onto)?\s*(?:the|our|your|a|another)?\s*"
    r"(?:next|new|second|third|fourth|another|following|last|final)\s+(?:one\s+)?"
    r"(?:lesson|topic|chapter)"
    r"|(?:next|new|second|third|fourth|another|last|final)\s+(?:lesson|topic|chapter)\s+"
    r"(?:is|was|will\s+be|would\s+be|is\s+going\s+to\s+be|for\s+today|today)"
    r"|(?:our|the|today'?s)\s+(?:next|new|second|third|fourth|last|final)\s+"
    r"(?:lesson|topic|chapter)"
    r"|\blesson\s+(?:number\s+)?(?:two|three|four|2|3|4)\b",
    re.I)

#: Original thresholds were 12 and 30 UTTERANCES; a turn pairs a tutor block with a student
#: block, so the turn-space equivalents are about half. `segmentation_report` is what says
#: whether that conversion held.
MIN_SEG = 6      # a boundary this close to the end cannot open a lesson
DEDUP = 15       # repeated announcements within this many turns are one boundary
ANN_ZONE = 8     # turns after a boundary in which the new lesson is named or read aloud


def lo_words(lo_text) -> set[str]:
    if not isinstance(lo_text, str) or not lo_text:
        return set()
    return {w.lower() for w in WORD.findall(lo_text) if len(w) > 2 and w.lower() not in STOPS}


def segment_session(tutor_text: list[str]) -> list[int]:
    """Segment boundaries as turn indices; 0 always first, len(turns) always last."""
    n = len(tutor_text)
    segs = [0]
    for i, t in enumerate(tutor_text):
        if t and BOUND.search(t) and i - segs[-1] >= DEDUP and n - i >= MIN_SEG:
            segs.append(i)
    return segs + [n]


def match_segment(segs: list[int], tutor_text: list[str], student_text: list[str],
                  lo_text: str) -> int:
    """Which segment handled this objective, or -1 when nothing separates them.

    Stage 1 looks only at the announcement zone just after each boundary, where a tutor names
    or reads out the new lesson. Stage 2 falls back to objective-word density over the whole
    segment. **A tie at either stage is not resolved arbitrarily** — it falls through to -1 and
    the caller uses the whole session, because guessing would put real exchanges under the
    wrong objective and that is the error this module exists to avoid.
    """
    nseg = len(segs) - 1
    if nseg < 2:
        return -1
    lw = lo_words(lo_text)
    if not lw:
        return -1
    scores = []
    for k in range(nseg):
        a, b = segs[k], min(segs[k + 1], segs[k] + ANN_ZONE)
        zone = {w.lower() for i in range(a, b) for w in WORD.findall(tutor_text[i] or "")}
        scores.append(len(lw & zone))
    top = max(scores)
    if top > 0 and scores.count(top) == 1:
        return int(np.argmax(scores))
    dens = []
    for k in range(nseg):
        a, b = segs[k], segs[k + 1]
        hits = sum(1 for i in range(a, b)
                   if lw & {w.lower() for w in WORD.findall(
                       (tutor_text[i] or "") + " " + (student_text[i] or ""))})
        dens.append(hits / max(b - a, 1))
    top = max(dens)
    if top > 0 and dens.count(top) == 1:
        return int(np.argmax(dens))
    return -1


def wrong_segment(k: int, nseg: int, key: str) -> int:
    """A DIFFERENT segment of the same session, picked by a hash of the row id.

    The placebo. If objective-localised features carry objective-specific signal, the matched
    segment must beat this; if the two score alike, the localisation is decorative and any
    within-session result built on it is an artifact.
    """
    if k < 0 or nseg < 2:
        return k
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    others = [j for j in range(nseg) if j != k]
    return others[h % len(others)]
