"""Small helpers shared by the scripts that write the paper's LaTeX tables."""
from __future__ import annotations

import re


def _caption_span(tex: str) -> tuple[int, int]:
    """Start and end of the `\\caption{...}` and the `\\label{...}` that follows it,
    together with a longtable row break (`\\\\`) if one closes them."""
    start = tex.index(r"\caption{")
    i, depth = start + len(r"\caption"), 0
    while True:
        c = tex[i]
        depth += c == "{"
        depth -= c == "}"
        i += 1
        if depth == 0:
            break
    m = re.compile(r"\s*\\label\{[^}]*\}(\\\\)?").match(tex, i)
    return start, (m.end() if m else i)


def caption_below(tex: str) -> str:
    """Move a table's caption (and its label) below the table.

    In a `table`/`table*` it goes just before `\\end{table}`. In a `longtable` it goes
    into an `\\endlastfoot`, so it prints once, under the last page of the table."""
    start, end = _caption_span(tex)
    cap = tex[start:end].strip()
    rest = tex[:start] + tex[end:].lstrip("\n")
    rest = re.sub(r"\n[ \t]*\n", "\n", rest, count=1) if rest[start - 1:start] == "\n" else rest
    if r"\begin{longtable}" in tex:
        if not cap.endswith("\\\\"):
            cap += "\\\\"
        foot = "\\bottomrule\n\\endfoot"
        assert foot in rest, "longtable without a \\bottomrule\\endfoot to extend"
        return rest.replace(foot, foot + "\n\\bottomrule\n" + cap + "\n\\endlastfoot", 1)
    k = max(rest.rfind(r"\end{table*}"), rest.rfind(r"\end{table}"))
    assert k >= 0, "no table environment"
    return rest[:k] + cap + "\n" + rest[k:]
