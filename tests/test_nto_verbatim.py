#!/usr/bin/env python
"""Assert every NTO tutor move reaches the annotation prompt **verbatim from the paper**.

This closes the one link `taxonomy_v3.verify_verbatim` cannot check. That function proves
*prompt == spec* (`docs/taxonomy_v3.md`); this file proves *spec == paper*, by holding Table 1
of the published taxonomy as literal text, transcribed from the PDF's text layer
(`pdftotext -layout`, arXiv:2603.05778, Zhou, Vanacore, Thompson, St John & Kizilcec 2026).

Why it matters beyond tidiness: a paraphrased definition can drop the very clauses that keep a
code from misfiling (a lost `NOT` clause makes a code fire far above its corpus rate). The paper claims the NTO codebook is used verbatim, so that claim is tested, not asserted.

Comparison normalises only whitespace and curly quotes — not words.

    python tests/test_nto_verbatim.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttapub import taxonomy_v3 as T  # noqa: E402

#: Table 1, verbatim. The bracketed editorial aside in the paper's
#: PROMPTING_ALTERNATIVE_REPRESENTATION row ("[Examples: … This will belong to the example
#: column.]") is a note to the authors' own future example column, not part of the definition,
#: and is excluded. Nothing else is altered.
PAPER_TABLE_1 = {
    # Tutoring support (5)
    "ASKING_TO_CLARIFY_CONTEXT":
        "Tutor asks the student about the problem to understand the context.",
    "PROBING_PRIOR_KNOWLEDGE":
        "Tutor probes about student's prior knowledge",
    "PROBING_UNDERSTAND":
        "Tutor questions whether student understood the statement made or step(s) done.",
    "CORRECTING_OWN_ERROR":
        "Tutor explicitly recognizes a mathematical error they made.",
    "STRATEGIZING":
        "Tutor discusses plans with the student to complete the problem.",
    # Learning support (15), paper order: higher -> lower student engagement
    "PROMPTING_RELATED_CONCEPTS":
        "Tutor prompts the student to make connections to related concepts without giving "
        "detailed explanations of the related concepts.",
    "PROMPTING_ALTERNATIVE_REPRESENTATION":
        "Tutor asks the student to represent the idea in a different form.",
    "PROMPTING_SELF_EXPLANATION":
        "Tutor prompts the student to explain actions or knowledge.",
    "PROMPTING_NEXT_STEP":
        "Tutor prompts the student to produce any step without adding new information.",
    "PROMPTING_SELF_CORRECTION":
        "Tutor prompts the student to correct their own mistakes without giving away the "
        "right way.",
    "FEEDBACK_CORRECT":
        "Tutor indicates that the student has provided a correct response.",
    "FEEDBACK_INCORRECT":
        "Tutor indicates that the student has provided an incorrect response.",
    "FEEDBACK_NEUTRAL":
        "Tutor comments without confirming accuracy or inaccuracy.",
    "REVOICING":
        "Tutor says what the student said in different ways.",
    "RESTATING":
        "Tutor says what the student said verbatim.",
    "GIVING_HINT":
        "Tutor provides a suggestion to guide the student toward a solution without revealing "
        "the answer.",
    "GIVING_EXAMPLE":
        "Tutor uses an example that is different from the student's problem.",
    "EXPLAINING_CONCEPTUAL":
        "Tutor gives instruction on concepts or facts.",
    "EXPLAINING_PROCEDURAL":
        "Tutor gives instruction about the algorithmic steps to solve the problem.",
    "GIVING_ANSWER":
        "Tutor provides the solution without including the solution steps.",
    # Social-emotional & motivational (7)
    "BUILDING_RAPPORT":
        "Tutor engages in relationship-building conversation, including when the tutor "
        "humanizes themselves.",
    "ENCOURAGING":
        "Tutor makes encouraging statements to students.",
    "ASKING_FEELING":
        "Tutor asks for the student's feelings.",
    "VALIDATING_FEELING":
        "Tutor validates the student's feelings.",
    "PRAISING_PROCESS":
        "Tutor praises the student's process of solving the problem.",
    "PRAISING_OUTCOME":
        "Tutor praises the student's outcome of student work.",
    "PRAISING_TRAITS":
        "Tutor praises the student's characteristic.",
    # Logistical (1)
    "TECHNICAL_ISSUES":
        "Tutor makes a comment related to technical problems of the tutoring context.",
}

#: Codes that are ours, not NTO's — they have no published wording to be verbatim against, so
#: they are listed here explicitly rather than inferred, and any new code forces a decision.
OURS = {
    # from the tutor interviews
    "RELEASE_HANDOFF", "TRANSFER_PROBE", "SIMPLIFYING_TO_SUBPROBLEM", "MOTIVATING_RELEVANCE",
    "SUMMARIZING_PROGRESS", "OFFERING_CHOICE", "NORMALIZING_DIFFICULTY",
    # from our own transcript review
    "GUIDING_SESSION", "EXPLAINING_TOOL", "WORKED_DEMONSTRATION",
    # residuals
    "TUTOR_UNINTELLIGIBLE", "TUTOR_OTHER", "TUTOR_NO_MOVE",
}


def norm(s: str) -> str:
    """Whitespace and quote characters only — never a word."""
    s = unicodedata.normalize("NFKD", s).replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    pool = T.tutor_pool()
    prompt = T.build_prompt("tutor")
    fail = []

    if len(PAPER_TABLE_1) != 28:
        fail.append(f"the transcribed table has {len(PAPER_TABLE_1)} rows, the paper has 28")

    for code, definition in PAPER_TABLE_1.items():
        if code not in pool:
            fail.append(f"{code}: in the paper, absent from our tutor pool")
            continue
        if norm(pool[code]["definition"]) != norm(definition):
            fail.append(f"{code}: definition differs from the paper\n"
                        f"    paper: {norm(definition)}\n"
                        f"    ours : {norm(pool[code]['definition'])}")
        if f"- {code}: {pool[code]['definition']}" not in prompt:
            fail.append(f"{code}: in the spec but not in the built prompt")

    unexplained = set(pool) - set(PAPER_TABLE_1) - OURS
    if unexplained:
        fail.append(f"codes in the pool that are neither NTO's nor declared ours: "
                    f"{sorted(unexplained)} — add them to OURS with their source, or fix "
                    f"the spec")
    orphan = OURS - set(pool)
    if orphan:
        fail.append(f"declared ours but no longer in the pool: {sorted(orphan)}")

    if fail:
        print("❌ NTO verbatim check FAILED\n")
        for f in fail:
            print("  " + f)
        return 1
    print(f"✅ all 28 NTO tutor moves are in the pool and verbatim in the built prompt "
          f"(arXiv:2603.05778 Table 1)\n"
          f"   tutor pool: {len(pool)} codes = 28 NTO + {len(OURS)} ours\n"
          f"   student pool: {len(T.codes('student'))} codes (ours; the paper has no student "
          f"moves)")
    return 0


# ── origins ─────────────────────────────────────────────────────────────────────────────────
#
# The paper's column table (`scripts/20_column_table.py`) states where every code came from,
# and it reads that from the codebook's own section layout rather than a second list. These
# counts pin that mapping: if a code moves between sections, or a student `src` cell is
# reworded past the markers in `_STUDENT_ORIGIN`, the table would quietly relabel a code's
# provenance in a published paper. That is the failure this guards.

EXPECTED_ORIGINS = {
    "tutor": {"NTO": 28, "INT": 7, "CLOSE": 6},
    "student": {"INT": 6, "CLOSE": 10},
}


def check_origins() -> list[str]:
    from collections import Counter

    fail = []
    for role, expected in EXPECTED_ORIGINS.items():
        got = dict(Counter(e["origin"] for e in T.pool(role).values()))
        if got != expected:
            fail.append(f"{role} origins {got}, expected {expected}")
    # the 28 NTO tutor codes must be exactly the ones transcribed from the paper
    nto = {c for c, e in T.pool("tutor").items() if e["origin"] == "NTO"}
    if nto != set(PAPER_TABLE_1):
        fail.append(f"codes marked NTO differ from the paper's Table 1: "
                    f"{sorted(nto ^ set(PAPER_TABLE_1))}")
    for role in ("tutor", "student"):
        for code in T.codes(role):
            if T.origin(code, role) not in T.ORIGIN_LABEL:
                fail.append(f"{role} {code} has unlabelled origin {T.origin(code, role)!r}")
    # residual is a property of the code, not an origin, and the counts are fixed by §2.6 and
    # the student table's RES markers
    n_res = {r: sum(T.is_residual(c, r) for c in T.codes(r)) for r in ("tutor", "student")}
    if n_res != {"tutor": 3, "student": 5}:
        fail.append(f"residual counts {n_res}, expected 3 tutor and 5 student")
    if any(T.is_residual(c, r) and T.origin(c, r) != "CLOSE"
           for r in ("tutor", "student") for c in T.codes(r)):
        fail.append("a residual is attributed outside close reading")
    return fail


def test_origins() -> None:
    fail = check_origins()
    assert not fail, "\n".join(fail)


if __name__ == "__main__":
    rc = main()
    fail = check_origins()
    if fail:
        print("\u274c origins:\n   " + "\n   ".join(fail))
        rc = 1
    else:
        print(f"\u2705 every code's origin parses from the codebook "
              f"({EXPECTED_ORIGINS['tutor']} tutor, {EXPECTED_ORIGINS['student']} student)")
    raise SystemExit(rc)
