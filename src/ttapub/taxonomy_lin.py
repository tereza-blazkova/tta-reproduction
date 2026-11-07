"""The Lin-scheme baseline's codebook, read from `docs/lin_scheme.md`, and its segmenter.

A baseline **using** the dialogue-act scheme of Lin et al. (2022), Table 2, applied by our LLM
annotator. It is not a replication: Lin's classifier and human-coded data are not public.

As with :mod:`ttapub.taxonomy_v3`, no definition is typed here. The prompt is built from the
markdown, and :func:`verify_verbatim` asserts each parsed definition survives into it.

The v3 code path does not import this module, and this module changes nothing in it.
"""
from __future__ import annotations

import re
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2] / "docs" / "lin_scheme.md"

#: Tags that receive a label but no feature column (`docs/lin_scheme.md` §4.4).
UNCODABLE = "UNCODABLE"

N_ROWS = 30                      # 29 of Lin's second-level tags + UNCODABLE
N_TUTOR_COLUMNS, N_STUDENT_COLUMNS = 23, 16

_ROLE_LETTERS = {"tutor": "T", "student": "S"}
_ROLE_WORD = {"tutor": "TUTOR", "student": "STUDENT"}


def _clean(s: str) -> str:
    s = re.sub(r"`([^`]*)`", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _scheme_block(text: str) -> str:
    start = text.index("## 3. The scheme")
    end = text.index("## 4. Decisions")
    return text[start:end]


def _parse() -> tuple[dict[str, dict], list[tuple[str, str]]]:
    block = _scheme_block(SPEC.read_text(encoding="utf-8"))
    tags: dict[str, dict] = {}
    rules: list[tuple[str, str]] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 7 and re.fullmatch(r"`[A-Z_]+`", cells[0]):
            tags[cells[0].strip("`")] = {
                "first": cells[1], "lin_name": cells[2], "role": cells[3], "example": cells[4],
                "definition": _clean(cells[5]), "source": _clean(cells[6])}
        elif len(cells) == 2 and " vs " in cells[0]:
            rules.append((_clean(cells[0]), _clean(cells[1])))
    if len(tags) != N_ROWS:
        raise RuntimeError(f"parsed {len(tags)} tags from {SPEC}, expected {N_ROWS}")
    if not rules:
        raise RuntimeError(f"parsed no boundary rules from {SPEC}")
    return tags, rules


_CACHE: dict[str, object] = {}


def _spec() -> tuple[dict[str, dict], list[tuple[str, str]]]:
    if "spec" not in _CACHE:
        _CACHE["spec"] = _parse()
    return _CACHE["spec"]  # type: ignore[return-value]


def tags() -> dict[str, dict]:
    return dict(_spec()[0])


def pool(role: str) -> dict[str, dict]:
    """Every tag the annotator may give this role, `UNCODABLE` included."""
    letter = _ROLE_LETTERS[role]
    return {c: e for c, e in tags().items() if letter in e["role"]}


def codes(role: str) -> list[str]:
    return list(pool(role))


def feature_codes(role: str) -> list[str]:
    """The tags that become columns: everything but `UNCODABLE`."""
    return [c for c in codes(role) if c != UNCODABLE]


def first_level(code: str) -> str:
    return tags()[code]["first"]


def first_level_groups(role: str) -> dict[str, list[str]]:
    """First-level tag -> this role's second-level tags under it."""
    out: dict[str, list[str]] = {}
    for c in feature_codes(role):
        out.setdefault(first_level(c), []).append(c)
    return out


def boundary_rules(role: str) -> list[tuple[str, str]]:
    """The rules whose two tags both exist for this role."""
    have = set(codes(role))
    return [(p, r) for p, r in _spec()[1] if set(re.findall(r"[A-Z_]{3,}", p)) <= have]


# ── segmentation ────────────────────────────────────────────────────────────────────────

#: A period after these does not end a sentence. Kept short on purpose: this is transcribed
#: speech, and the list only has to cover what the transcripts' punctuation actually produces.
_ABBREV = ("mr", "mrs", "ms", "dr", "miss", "sir", "e.g", "i.e", "vs", "etc")
_BOUNDARY = re.compile(r"(?<=[.?!])[\"')\]]*\s+")
#: A fragment with no words, only `[unclear]` markers. Pilot round 1: these made up 8.5% of student
#: "sentences". A marker is a gap inside what the speaker said, not an act of its own.
_BARE = re.compile(r"\W*(\[\s*unclear\s*\]\W*)+", re.I)


def segment(text: str) -> list[str]:
    """One speaker's text -> sentences, split after `.`, `?` or `!` followed by whitespace.

    Decimals survive because no whitespace follows their point. A fragment of bare `[unclear]`
    markers joins the sentence before it, or the one after it when it comes first, and stands
    alone only when the speaker's whole text is markers. Nothing is dropped or reworded: joining
    the result with spaces gives back the input's words.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        left = text[start:m.start()].rstrip()
        last = left.rsplit(" ", 1)[-1].lower().rstrip(".")
        if last in _ABBREV:
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    merged: list[str] = []
    pending = ""
    for sent in out:
        if _BARE.fullmatch(sent):
            if merged:
                merged[-1] = f"{merged[-1]} {sent}"
            else:
                pending = f"{pending} {sent}".strip()
        else:
            merged.append(f"{pending} {sent}".strip())
            pending = ""
    return merged or ([pending] if pending else [])


# ── prompt ──────────────────────────────────────────────────────────────────────────────

_SHARED_RULES = """# How to choose one tag
1. Label what the {unit} MAINLY does. There is NO priority order between tags: do not prefer a
   tag because it appears earlier in the list.
2. A QUESTION MARK DOES NOT DECIDE THE TAG. Classify by what the {unit} does.
3. Use the other speaker's words and the surrounding {unit}s to decide what this one is doing.
4. These are ASR transcripts of spoken sessions, so expect disfluency, [unclear] markers,
   [Speaker:...] markers and short backchannels. Ignore the markers and judge what the speaker
   is doing from the words that remain, however unpolished.
5. UNCODABLE is only for a {unit} with nothing intelligible. A {unit} with an [unclear] marker
   and readable words is tagged on those words. UNCODABLE is never a way to avoid a hard
   decision."""


def _format_pool(p: dict[str, dict]) -> str:
    lines = []
    for code, e in p.items():
        lines.append(f"- {code}: {e['definition']}")
        lines.append(f"    example: {e['example']}")
    return "\n".join(lines)


def build_prompt(role: str, unit: str = "sentence") -> str:
    if role not in _ROLE_WORD:
        raise ValueError(f"role must be tutor or student, got {role!r}")
    if unit not in ("sentence", "turn"):
        raise ValueError(f"unit must be sentence or turn, got {unit!r}")
    p = pool(role)
    r = _ROLE_WORD[role]
    rules = "\n".join(f"- {pair}: {rule}" for pair, rule in boundary_rules(role))
    return f"""You are an expert educational discourse analyst. Assign a dialogue-act tag to each numbered {r} {unit} in a 1:1 maths tutoring session. Assign EXACTLY ONE tag per {unit}. Return ONLY a JSON array — no outer object, no explanation.

# Allowed tags
{", ".join(p)}

# Definitions
{_format_pool(p)}

{_SHARED_RULES.format(unit=unit)}

# Boundary rules
{rules}

Output format: [{{"n": <int>, "code": "<TAG>"}}, ...]"""


def schema(role: str) -> dict:
    """Guided-JSON schema: the tag vocabulary is enforced at decode time."""
    return {"type": "array",
            "items": {"type": "object",
                      "properties": {"n": {"type": "integer"},
                                     "code": {"type": "string", "enum": codes(role)}},
                      "required": ["n", "code"], "additionalProperties": False}}


def verify_verbatim(role: str, prompt: str | None = None) -> int:
    """Assert every parsed definition reaches the prompt unchanged. Returns the number checked."""
    prompt = build_prompt(role) if prompt is None else prompt
    checked = 0
    for code, e in pool(role).items():
        if f"- {code}: {e['definition']}" not in prompt:
            raise AssertionError(f"definition altered for {code}:\n  {e['definition']}")
        checked += 1
    return checked
