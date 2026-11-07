"""Annotation taxonomy **v3** — the 57-code codebook, read from its spec.

41 tutor codes + 16 student codes = the 57 conversation columns of **clean57**.

**No definition is ever re-typed here.** Every definition is parsed out of
`docs/taxonomy_v3.md` at prompt-build time, and :func:`verify_verbatim` asserts that each
parsed definition appears in the built prompt unchanged. This is not ceremony: a
definition re-typed as a one-liner can lose a `NOT` clause, and the code then fires far above
its corpus rate. The spec is the single source of truth; editing a definition means editing the markdown.

Pools, per the spec:

* **tutor (41)** — all 28 published NTO codes (wording verbatim: tutoring support 5, learning
  support 15, social-emotional 7, logistical 1), our 7 interview codes, 3 data-derived codes,
  3 residuals.
* **student (16)** — the flat single-label scheme: two reasoning codes, two answer codes by
  confidence, two question codes, affect, off-task, and 5 residuals.

**One label per turn, chosen by primary function, with no priority order.** A global ranking
would decide every overlap in favour of whichever codes sat on top and build our assumptions
into the counts; instead the prompt carries the spec's pairwise boundary rules (§3). The
annotator may record an optional `also` code when a turn genuinely serves two functions: it is
never used as a label, only to measure which pairs still need a rule (pilot, §3.3).
"""
from __future__ import annotations

import re
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2] / "docs" / "taxonomy_v3.md"

#: Spec sections that define the tutor pool, in prompt order.
_TUTOR_SECTIONS = ["### 2.1", "### 2.2", "### 2.3", "### 2.4", "### 2.5", "### 2.6"]
_STUDENT_SECTION = "## 4."
_BOUNDARY_SECTION = "## 3."

N_TUTOR, N_STUDENT = 41, 16

#: Where each tutor section's codes came from. The spec groups the tutor pool by provenance
#: already (§2.1-2.3 are the published taxonomy, §2.4 the interviews, §2.5 our transcript
#: review, §2.6 bookkeeping), so the section *is* the origin and nothing has to be restated.
_TUTOR_ORIGIN = {"### 2.1": "NTO", "### 2.2": "NTO", "### 2.3": "NTO",
                 "### 2.4": "INT", "### 2.5": "CLOSE", "### 2.6": "CLOSE"}

#: Sections and `src` markers whose codes are residuals. Residual is a property of the *code*
#: -- it carries no pedagogical claim -- and not of where the code came from, which is why it
#: is tracked separately from the origin rather than as a value of it.
_RESIDUAL_TUTOR_SECTION = "### 2.6"

#: The student pool is one flat table with a free-text `src` cell, so its origin is read from
#: that cell. Order matters: "INT" is tested before the prior-codebook markers because an entry
#: may name both ("INT §5.3 split of REASONING_SHOWN" is an interview code).
_STUDENT_ORIGIN = ((("INT",), "INT"),
                   (("DATA",), "CLOSE"),
                   (("new",), "CLOSE"),
                   (("RES",), "CLOSE"))

#: Human-readable origins, for a table in the paper.
#: Three origins, covering all 57 codes. A code is published, or it came from asking tutors,
#: or it came from reading transcripts closely -- which includes the codes a validation round
#: adds, since checking labels against the turns they describe is reading transcripts.
#:
#: The residuals belong to the third. They were introduced during human annotation, because
#: without them turns with no pedagogical content were being absorbed by substantive codes.
#: They are a fix for a measurement failure someone observed, which is
#: close reading, and they protect every other code's rate from it.
ORIGIN_LABEL = {
    "NTO": "Published NTO taxonomy, wording verbatim",
    "INT": "Tutor interview study",
    "CLOSE": "Close reading of transcripts",
}


# ── spec parsing ────────────────────────────────────────────────────────────────────────

def _section(text: str, start: str) -> str:
    """The spec text from heading `start` up to the next heading of the same depth or higher."""
    i = text.index(start)
    depth = len(start.split(" ")[0])          # "###" -> 3
    rest = text[i + len(start):]
    for m in re.finditer(r"^(#{1,6})\s", rest, re.M):
        if len(m.group(1)) <= depth:
            return rest[: m.start()]
    return rest


def _rows(block: str) -> list[list[str]]:
    """Markdown table rows whose first cell is a `BACKTICKED_CODE`."""
    out = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and re.fullmatch(r"`[A-Z_]+`", cells[0]):
            out.append([cells[0].strip("`")] + cells[1:])
    return out


def _clean(s: str) -> str:
    """Strip markdown emphasis and backticks from a definition, keep the words."""
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]*)\*", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse() -> tuple[dict[str, dict], dict[str, dict], list[tuple[str, str]]]:
    text = SPEC.read_text(encoding="utf-8")

    tutor: dict[str, dict] = {}
    for sec in _TUTOR_SECTIONS:
        block = _section(text, sec)
        # §2.4 carries (definition, interview, distinct from); §2.5 (definition, distinct from)
        distinct_at = 3 if sec == "### 2.4" else (2 if sec == "### 2.5" else None)
        for cells in _rows(block):
            code, definition = cells[0], _clean(cells[1])
            entry = {"definition": definition, "section": sec.strip("# "),
                     "origin": _TUTOR_ORIGIN[sec],
                     "residual": sec == _RESIDUAL_TUTOR_SECTION}
            if distinct_at is not None and len(cells) > distinct_at:
                entry["distinct_from"] = _clean(cells[distinct_at])
            tutor[code] = entry

    student: dict[str, dict] = {}
    for cells in _rows(_section(text, _STUDENT_SECTION)):
        src = _clean(cells[2]) if len(cells) > 2 else ""
        origin = next((o for markers, o in _STUDENT_ORIGIN
                       if any(m in src for m in markers)), None)
        if origin is None:
            raise RuntimeError(f"student code {cells[0]} has src {src!r}, which matches no "
                               f"origin in _STUDENT_ORIGIN — add it there rather than "
                               f"defaulting, or the paper's column table will mislabel it")
        student[cells[0]] = {"definition": _clean(cells[1]), "section": "4",
                             "origin": origin, "src": src, "residual": src.strip() == "RES"}

    boundary: list[tuple[str, str]] = []
    for line in _section(text, _BOUNDARY_SECTION).splitlines():
        line = line.strip()          # the §3 table is indented inside a numbered list
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 2 and "`" in cells[0] and cells[0] != "pair":
            boundary.append((_clean(cells[0]), _clean(cells[1])))

    if len(tutor) != N_TUTOR:
        raise RuntimeError(f"parsed {len(tutor)} tutor codes from {SPEC}, expected {N_TUTOR}")
    if len(student) != N_STUDENT:
        raise RuntimeError(f"parsed {len(student)} student codes, expected {N_STUDENT}")
    if not boundary:
        raise RuntimeError(f"parsed no boundary rules from {SPEC} §3")
    missing = [c for p in (tutor, student) for c, e in p.items() if not e.get("origin")]
    if missing:
        raise RuntimeError(f"no origin parsed for {missing}")
    return tutor, student, boundary


_CACHE: dict[str, object] = {}


def _spec() -> tuple[dict[str, dict], dict[str, dict], list[tuple[str, str]]]:
    if "spec" not in _CACHE:
        _CACHE["spec"] = _parse()
    return _CACHE["spec"]  # type: ignore[return-value]


def tutor_pool() -> dict[str, dict]:
    return dict(_spec()[0])


def student_pool() -> dict[str, dict]:
    return dict(_spec()[1])


def pool(role: str) -> dict[str, dict]:
    return tutor_pool() if role == "tutor" else student_pool()


def codes(role: str) -> list[str]:
    return list(pool(role))


def all_codes() -> list[str]:
    return codes("tutor") + codes("student")


def boundary_rules() -> list[tuple[str, str]]:
    return list(_spec()[2])


# ── prompt construction ─────────────────────────────────────────────────────────────────

_ROLE_WORD = {"tutor": "TUTOR", "student": "STUDENT"}

_SHARED_RULES = """# How to choose one code
1. Label what the turn is MAINLY for — its primary function. There is NO priority order
   between codes: do not prefer a code because it appears earlier in the list.
2. A QUESTION MARK NEVER DECIDES THE LABEL. Classify by what the turn DOES, not by its
   grammatical form.
3. Use a residual code only after rejecting every substantive code. {R}_OTHER means
   "this turn matters but the list is missing it"; {R}_NO_MOVE means "nothing notable here".
   {R}_NO_MOVE is the fallback, never a way to avoid a hard decision.
4. These are ASR transcripts of spoken sessions, so expect disfluency, [unclear] markers and
   short backchannels. Judge what the speaker is doing, not how polished it sounds."""

_ALSO_RULE = """# Optional second code
If a turn genuinely serves two functions, put the second one in "also". It is NOT a label and
is never counted as one — it only records which code pairs are hard to separate. Leave it out
whenever one code clearly fits."""


def _format_pool(p: dict[str, dict]) -> str:
    lines = []
    for code, e in p.items():
        lines.append(f"- {code}: {e['definition']}")
        if e.get("distinct_from"):
            lines.append(f"    distinct from: {e['distinct_from']}")
    return "\n".join(lines)


def _format_boundaries(role: str) -> str:
    """The spec's §3 pairwise rules; student-side rules are those naming a student code."""
    stu = set(codes("student"))
    out = []
    for pair, rule in boundary_rules():
        names = set(re.findall(r"[A-Z_]{3,}", pair))
        is_student = bool(names & stu)
        if (role == "student") == is_student:
            out.append(f"- {pair}: {rule}")
    return "\n".join(out)


def build_prompt(role: str, also: bool = True) -> str:
    """The full system prompt for one pool. `also=False` drops the second-code field."""
    if role not in _ROLE_WORD:
        raise ValueError(f"role must be tutor or student, got {role!r}")
    p = pool(role)
    r = _ROLE_WORD[role]
    also_block = f"\n\n{_ALSO_RULE}" if also else ""
    also_field = ', "also": "<CODE or null>"' if also else ""
    return f"""You are an expert educational discourse analyst. Classify each {r} turn in a 1:1 maths tutoring session. Assign EXACTLY ONE code per turn. Return ONLY a JSON array — no outer object, no explanation.

# Allowed codes
{", ".join(p)}

# Definitions
{_format_pool(p)}

{_SHARED_RULES.format(R=r)}

# Boundary rules
{_format_boundaries(role)}{also_block}

Output format: [{{"turn": <int>, "code": "<CODE>"{also_field}}}, ...]"""


def schema(role: str, also: bool = True, null_also: bool = True) -> dict:
    """Guided-JSON schema for one pool: the label vocabulary is enforced at decode time.

    `null_also=False` swaps the nullable union for a plain enum with a NONE sentinel, for
    grammar backends that will not compile a union (the data is identical either way).
    """
    c = codes(role)
    props: dict[str, dict] = {"turn": {"type": "integer"},
                              "code": {"type": "string", "enum": c}}
    if also:
        props["also"] = ({"anyOf": [{"type": "string", "enum": c}, {"type": "null"}]}
                         if null_also else {"type": "string", "enum": c + [NONE_SENTINEL]})
    return {"type": "array",
            "items": {"type": "object", "properties": props,
                      "required": ["turn", "code"], "additionalProperties": False}}


NONE_SENTINEL = "NONE"


def verify_verbatim(role: str, prompt: str | None = None) -> int:
    """Assert every parsed definition survives into the built prompt unchanged.

    Returns the number of definitions checked. Called before every annotation run — a run
    whose prompt has drifted from the spec is not comparable to anything.
    """
    prompt = build_prompt(role) if prompt is None else prompt
    checked = 0
    for code, e in pool(role).items():
        if f"- {code}: {e['definition']}" not in prompt:
            raise AssertionError(f"definition altered for {code}:\n  {e['definition']}")
        checked += 1
    return checked




if __name__ == "__main__":
    import sys
    role = sys.argv[1] if len(sys.argv) > 1 else "tutor"
    print(build_prompt(role))
    print(f"\n--- {len(codes(role))} codes, {verify_verbatim(role)} definitions verified "
          f"verbatim against {SPEC.name} ---", file=sys.stderr)


def origin(code: str, role: str) -> str:
    """The provenance key of one code: NTO, INT or CLOSE."""
    return pool(role)[code]["origin"]


def is_residual(code: str, role: str) -> bool:
    """Whether a code is a residual: it absorbs turns rather than asserting a construct.

    Orthogonal to :func:`origin`. A residual's rate is still a number the model reads, and
    `STUDENT_OTHER` has carried a real finding (it is mostly tutor speech mis-tagged as
    student, so it doubles as a diarisation-error detector) -- but no pedagogical claim may
    rest on one.
    """
    return pool(role)[code]["residual"]
