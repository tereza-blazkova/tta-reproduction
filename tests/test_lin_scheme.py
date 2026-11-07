#!/usr/bin/env python
"""Check `docs/lin_scheme.md` against Lin et al. (2022) Table 2, the prompt, and the segmenter.

`TABLE_2` is the table as read from the rendered page, because the PDF's text layer garbles the
merged Role cells. Lin's own feature counts (18 first-level, 41 second-level, §3.5) are an
independent check on the role mapping, so a misread role fails here.

    python tests/test_lin_scheme.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttapub import taxonomy_lin as L  # noqa: E402

#: first-level tag -> [(Lin's second-level name, role, example)], in Table 2's order.
TABLE_2 = {
    "HINT": [("Information", "T", "It can be any one of the cards in the deck."),
             ("Hint by Image", "T", "[Image]"),
             ("Observation", "T&S", "We have 80.")],
    "DIRECTIVE": [("Directive", "T", "Check this definition.")],
    "ACKNOWLEDGE": [("Acknowledge", "T&S", "Alright!")],
    "REQUEST_CONFIRMATION": [("Evaluation Question", "T", "Does that make sense?")],
    "REQUEST_FEEDBACK": [("Request Feedback by Image", "S", "[Image]"),
                         ("Confirmation Question", "S", "Would the answer be 30?")],
    "POSITIVE_FEEDBACK": [("General Positive Feedback", "T", "Correct!"),
                          ("Elaborated Positive Feedback", "T",
                           "Your formula for period is correct!")],
    "NEGATIVE_FEEDBACK": [("Negative Feedback", "T", "No, it is incorrect.")],
    "LUKEWARM_FEEDBACK": [("Lukewarm Feedback", "T", "Almost correct, but the sign is missing.")],
    "CORRECTION": [("Correction", "T&S", "*We will")],
    "QUESTION": [("Direction Question", "S", "How do I do that?"),
                 ("Information Question", "S", "What are the units for W?"),
                 ("Probing Question", "T", "How many options can it be?"),
                 ("Open Question", "T", "What do you think we could try next?"),
                 ("Factual Question", "T", "What is the value of x?"),
                 ("Operational Question", "T", "Any questions on this?"),
                 ("Ready Question", "T", "Are you ready to begin?"),
                 ("Extra Domain Question", "T&S", "How are you doing today?")],
    "ANSWER": [("Yes-No Answer", "T&S", "Yes, that would be very helpful."),
               ("WH Answer", "T&S", "It is 6."),
               ("Ready Answer", "S", "Yes, I'm ready."),
               ("Extra Domain Answer", "T&S", "I'm good.")],
    "STATEMENT": [("Explanation", "T&S", "The straight line is the line on the bottom."),
                  ("Greeting", "T&S", "Hello!"),
                  ("Extra Domain Other", "T&S", "Welcome to use this app!"),
                  ("Reassurance", "T", "No problem, I will help you."),
                  ("Understanding", "S", "Ok, got it."),
                  ("Not Understanding", "S", "I don't know why.")],
}
IMAGE_TAGS = {"Hint by Image", "Request Feedback by Image"}


def test_table_counts() -> None:
    rows = [r for v in TABLE_2.values() for r in v]
    assert len(TABLE_2) == 12 and len(rows) == 31
    # Lin §3.5: 18 first-level and 41 second-level features, 6 and 10 of them shared by role.
    shared2 = [r for r in rows if r[1] == "T&S"]
    assert len(shared2) == 10 and len(rows) + len(shared2) == 41
    shared1 = [k for k, v in TABLE_2.items()
               if {"T", "S"} <= {c for r in v for c in r[1].split("&")}]
    assert len(shared1) == 6 and len(TABLE_2) + len(shared1) == 18


def test_doc_matches_table() -> None:
    got = {e["lin_name"]: e for c, e in L.tags().items() if c != L.UNCODABLE}
    want = {r[0]: (k, r[1], r[2]) for k, v in TABLE_2.items() for r in v
            if r[0] not in IMAGE_TAGS}
    assert set(got) == set(want), set(got) ^ set(want)
    for name, (first, role, example) in want.items():
        e = got[name]
        assert (e["first"], e["role"], e["example"]) == (first, role, example), name


def test_pools() -> None:
    assert len(L.feature_codes("tutor")) == L.N_TUTOR_COLUMNS
    assert len(L.feature_codes("student")) == L.N_STUDENT_COLUMNS
    for role in ("tutor", "student"):
        assert L.UNCODABLE in L.codes(role) and L.UNCODABLE not in L.feature_codes(role)
        assert all("image" not in c.lower() for c in L.codes(role))


def test_prompt() -> None:
    for role in ("tutor", "student"):
        for unit in ("sentence", "turn"):
            p = L.build_prompt(role, unit)
            assert L.verify_verbatim(role, p) == len(L.codes(role))
            other = "student" if role == "tutor" else "tutor"
            for c in set(L.codes(other)) - set(L.codes(role)):
                assert f"- {c}:" not in p, f"{c} leaked into the {role} prompt"
        assert L.schema(role)["items"]["properties"]["code"]["enum"] == L.codes(role)
    # a rule naming a tag this role cannot have would confuse the annotator
    assert all("GENERAL_POSITIVE_FEEDBACK" not in pair for pair, _ in L.boundary_rules("student"))


def test_every_tag_states_its_source() -> None:
    for code, e in L.tags().items():
        assert e["source"], code
        assert any(k in e["source"] for k in ("OURS", "M13", "L22-P")), code


def test_segment() -> None:
    cases = {
        "": [],
        "Good.": ["Good."],
        "It is 3.5 metres. Why?": ["It is 3.5 metres.", "Why?"],
        "Hello Mr. Smith. Okay!": ["Hello Mr. Smith.", "Okay!"],
        "No end punctuation": ["No end punctuation"],
        "Yes. [unclear]": ["Yes. [unclear]"],
        "[unclear]": ["[unclear]"],
        "[unclear] [UNCLEAR]": ["[unclear] [UNCLEAR]"],
        "Why? [unclear] Okay.": ["Why?", "[unclear] Okay."],
        "[unclear]. Okay. Yes. [unclear]. [UNCLEAR]": ["[unclear]. Okay.", "Yes. [unclear]. [UNCLEAR]"],
        "I chose C because it's over... [unclear] Yeah.":
            ["I chose C because it's over...", "[unclear] Yeah."],
        "  extra   spaces.  Fine. ": ["extra spaces.", "Fine."],
    }
    for text, want in cases.items():
        got = L.segment(text)
        assert got == want, (text, got)
        assert " ".join(got).split() == text.split(), text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
