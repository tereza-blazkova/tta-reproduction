#!/usr/bin/env bash
# Download the foundation-model weights the GPU baselines read. The models load with
# local_files_only=True, so this runs once, before any GPU row, with network access.
#
#   HF_HOME=/path/to/cache PY=.venv-gpu-b/bin/python bash reproduce/download_weights.sh
#
# Then run the GPU rows with the same HF_HOME and HF_HUB_OFFLINE=1. The TabPFN repositories ask
# you to accept Prior Labs' licence on huggingface.co first; log in with `hf auth login` if a
# download is refused.
set -euo pipefail
PY=${PY:-python}
"$PY" - <<'PYEOF'
from huggingface_hub import hf_hub_download
files = [
    ("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-20260909.safetensors", None),
]
for repo, name, rev in files:
    path = hf_hub_download(repo, name, revision=rev)
    print(f"ok  {repo}:{name}  ->  {path}")
PYEOF
