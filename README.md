# The Importance of Student Signal in Predicting Tutoring Outcomes

Code for the paper's analyses. It runs on the *Trace the Ace* competition data, which is
distributed by the National Tutoring Observatory and is not included here.

```
src/ttapub/    shared code: data loading, codebook, features, models, metrics
scripts/       one script per step, numbered in order
figures/       the paper's figures
docs/          the annotation codebook (the prompts are built from it)
data/folds/    the cross-validation folds
reproduce/     every run in the paper, with its command, and a script that runs them all
env/           exact package versions
```

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Put the competition files in `data/raw/` under the names listed in `data/README.md`, then check
them with `.venv/bin/python scripts/00_check_data.py`.

## Run

1. **Annotate.** Every turn is labelled by `Qwen/Qwen3.5-122B-A10B-FP8` through vLLM, on GPUs:
   `sh scripts/launch_shards.sh <n_gpus> full 0 "" "train test"`.
2. **Everything else.** `bash reproduce/run_all.sh` runs every analysis in the paper in order.
   `reproduce/runs.tsv` lists each run's command. The TabPFN baseline needs a GPU
   (`env/gpu-*.txt`, weights via `reproduce/download_weights.sh`); add `--cpu-only` to skip it.
3. **Figures.** `bash figures/build_all.sh`.

Results match the paper's to floating-point precision with the package versions in `env/` and
the authors' labels, except the TabPFN runs, whose GPU kernels are not deterministic. A new
annotation gives close but not identical numbers.

## License

MIT. The codebook quotes the National Tutoring Observatory's published tutor-move definitions,
which remain under their authors' terms.
