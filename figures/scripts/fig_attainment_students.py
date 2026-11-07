#!/usr/bin/env python
"""fig_attainment_students — which codes mark a higher-attaining student (Fig. 4).

One plot per role, drawn by `fig_code_occurrence` without its attainment-model plot:
  a  student signals: how much more common each is in higher- than in lower-attaining students'
     sessions (stage 48).
  b  tutor moves, likewise.

The code plots with what each code adds to the attainment model are the SI figure drawn by
`fig_codes_by_attainment.py`. Drawn at the text width and printed 1:1. Reads only `figures/input/`.

    python figures/scripts/fig_attainment_students.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import fig_code_occurrence as occ
import style as st

W = st.TEXT_W                           # the text width: prints 1:1
#: the shared panel functions' marks are sized for wider canvases
MARK = 0.82
OCC_ROW = 0.091                         # inches per code
OCC_LABEL_W = {"student": 1.37, "tutor": 1.44}  # the code labels: the longest are bold, with a SHAP rank
OCC_GAP = 0.20                          # between the two blocks
#: the occurrence plots share what the labels leave of the text width
OCC_W = (W - 0.10 - sum(OCC_LABEL_W.values()) - 2 * 0.05 - OCC_GAP) / 2   # 0.10: room for b's "higher attainment"
OCC_AXIS, OCC_HEAD = 0.33, 0.30         # under a plot (axis labels), over it (title, direction labels)


def main() -> None:
    st.apply_style(paper=True, mark=MARK)
    shown, add, rows = occ.occurrence_data()
    span = max(rows.values())                            # the shorter block centres on the taller
    h = span * OCC_ROW
    b_row = OCC_AXIS
    H = b_row + h + OCC_HEAD
    fig = plt.figure(figsize=(W * st.SCALE, H * st.SCALE))
    x_tutor = OCC_LABEL_W["student"] + 0.05 + OCC_W + OCC_GAP
    for role, letter, title, left in (("student", "a", "Student signals", 0.0), ("tutor", "b", "Tutor moves", x_tutor)):
        occ.occurrence_block(fig, W, H, role, letter, title, left, OCC_LABEL_W[role], b_row, span, shown, add,
                             rows, occ_w=OCC_W, row=OCC_ROW, label_fs=st.tsize("code"), show_add=False)
    st.save(fig, "fig_attainment_students", width=W)


if __name__ == "__main__":
    main()
