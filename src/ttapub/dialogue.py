"""Transcript -> paired turns, and the text each distilled child reads.

**Turn pairing (convention A).** Consecutive utterances by the same speaker are joined, and a
turn is one `(teacher, student)` exchange: the tutor's utterance plus the student's reply to it.
Numbering starts at 0 if the student opens the session and at 1 otherwise. Every annotation
is joined on this numbering. Do not "fix" it.

**Every paired turn is kept.** The codebook
codes every tutor turn and every student turn, so :func:`build_turns` keeps every paired turn
and flags `has_teacher` / `has_student`. Tutor codes are computed on `has_teacher` turns and
student codes on `has_student` turns; mixing the axes would put a tutor rate's denominator on
the student's turns.

**Child input text, frozen.** `tutor_text` is the previous student turn + `" [T] "` + the
tutor utterance; `student_text` is the tutor utterance + `" [S] "` + the student turn. The
truncations are fixed. Both sides are available at
inference, so nothing here leaks.
"""
from __future__ import annotations

import pandas as pd

#: Character truncations, fixed in advance.
TRUNC = {"tutor_ctx": 200, "tutor_utt": 220, "student_ctx": 220, "student_utt": 300,
         "table": 220}


def _add(acc: str, content: str) -> str:
    content = content.strip()
    if not content:
        return acc
    return f"{acc} {content}".strip() if acc else content


def build_turns(tx: pd.DataFrame) -> list[dict]:
    """One transcript -> [{turn, teacher, student}, ...]; `background` utterances dropped."""
    tx = tx.sort_values("utterance_id")
    utts: list[tuple[str, str]] = []
    for _, u in tx.iterrows():
        role = str(u["role"]).strip().lower()
        content = "" if pd.isna(u["content"]) else str(u["content"]).strip()
        if not content or role not in ("tutor", "student"):
            continue
        utts.append(("teacher" if role == "tutor" else "student", content))
    if not utts:
        return []

    cur_role = utts[0][0]
    cur = {"turn": 0 if cur_role == "student" else 1, "teacher": "", "student": ""}
    out: list[dict] = []
    for role, content in utts:
        if role == "teacher" and cur_role == "student":
            out.append(cur)
            cur = {"turn": cur["turn"] + 1, "teacher": "", "student": ""}
        cur_role = role
        cur[role] = _add(cur[role], content)
    # the final partial turn counts whenever it carries anything at all, so a trailing tutor
    # turn is never dropped
    if cur["teacher"] or cur["student"]:
        out.append(cur)
    return out


def tutor_text(turns: list[dict], i: int) -> str:
    """Text the tutor-side child reads for turn i: previous student turn + the utterance."""
    prev = turns[i - 1]["student"] if i > 0 else ""
    return f"{prev[:TRUNC['table']]} [T] {turns[i]['teacher'][-TRUNC['tutor_utt']:]}"


def student_text(turns: list[dict], i: int) -> str:
    """Text the student-side child reads for turn i: the tutor's utterance + the reply."""
    t = turns[i]
    return (f"{t['teacher'][-TRUNC['tutor_ctx']:]} [S] "
            f"{t['student'][:TRUNC['student_utt']]}")


def turn_frame(session_id: str, turns: list[dict]) -> pd.DataFrame:
    """The per-turn row a turn table holds: ids, position, both child texts, both flags."""
    n = len(turns)
    rows = []
    for i, t in enumerate(turns):
        rows.append({
            "session_id": session_id,
            "turn": t["turn"],
            "pos": i,
            "n_turns": n,
            "has_teacher": bool(t["teacher"].strip()),
            "has_student": bool(t["student"].strip()),
            "tutor_text": tutor_text(turns, i),
            "student_text": student_text(turns, i),
        })
    return pd.DataFrame(rows)
