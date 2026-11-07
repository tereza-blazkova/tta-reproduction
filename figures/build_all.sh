#!/usr/bin/env bash
# Every figure in the paper: prepare the inputs from results/, then draw.
set -euo pipefail
cd "$(dirname "$0")/scripts"
mkdir -p ../input ../output
PY=${PY:-python3}
PREPARE=(prepare_presence_inputs prepare_timing_inputs prepare_session_length_inputs prepare_student_attainment_inputs prepare_sessions_inputs prepare_main_figure_inputs prepare_mechanisms_figure_inputs prepare_beeswarm_inputs prepare_rq1_figure_inputs prepare_explain_inputs prepare_block_sequences_inputs prepare_student_heterogeneity_inputs prepare_head_to_head_inputs prepare_attainment_incremental_inputs prepare_code_occurrence_inputs prepare_learner_attribution_inputs prepare_attainment_blocks_inputs prepare_recalibration_intervals_inputs)
FIGURES=(fig_codes fig_within_student fig_attainment_students fig_codes_add_attainment fig_main_measurement fig_tabpfn_shap_si fig_bystate fig_attainment_si fig_attainment_blocks)
for s in "${PREPARE[@]}"; do echo "== $s"; "$PY" "$s.py" > /dev/null; done
for s in "${FIGURES[@]}"; do echo "== $s"; "$PY" "$s.py" > /dev/null; done
echo "== fig_blocks --table"; "$PY" fig_blocks.py --table > /dev/null
echo "done: figures/output/"
