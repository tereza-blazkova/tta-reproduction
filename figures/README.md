# Figures

`bash figures/build_all.sh` rebuilds every figure in the paper from `results/`.
Each `prepare_*_inputs.py` reduces `results/` to small CSVs in `figures/input/`, and each
`fig_*.py` reads only those and writes `figures/output/`. `style.py` sets fonts and colours.

| output | drawn by (`figures/scripts/`) |
|---|---|
| `fig_codes.pdf` | `fig_codes.py` |
| `fig_blocks_table.pdf` | `fig_blocks.py --table` |
| `fig_within_student.pdf` | `fig_within_student.py` |
| `fig_attainment_students.pdf` | `fig_attainment_students.py` |
| `fig_codes_add_attainment.pdf` | `fig_codes_add_attainment.py` |
| `fig_measurement_si.pdf` | `fig_main_measurement.py` |
| `fig_tabpfn_shap_si.pdf` | `fig_tabpfn_shap_si.py` |
| `fig_bystate_pair.pdf` | `fig_bystate.py` |
| `fig_attainment_si.pdf` | `fig_attainment_si.py` |
| `fig_attainment_blocks.pdf` | `fig_attainment_blocks.py` |
