# spectral_paper_cluster

Simple starter setup for clustering experiments with PyTorch.

## Structure

- `configs/` experiment configs
- `data/raw/` raw datasets (kept with placeholder `.gitkeep`)
- `data/processed/` processed artifacts (kepr with placeholder `.gitkeep`)
- `notebooks/` exploration notebooks
- `experiments/` runnable experiment pipelines (one file per experiment)
- `src/clustering/` clustering algorithms and models
- `src/features/` text feature extraction modules
- `src/evaluation/` clustering metrics and reports
- `src/pipelines/` end-to-end experiment orchestration
- `outputs/` plots and run artifacts (kept with placeholder `.gitkeep`)

Result plot will be written to `outputs/clusters.png`.

## Run Pattern

- Put each executable experiment in `experiments/` (for example `experiments/tfidf_kmeans.py`).
- Keep reusable code in `src/` and call it from experiment files.
- Use `configs/` to externalize parameters.


## Experiments
Siehe die Experimentbeschreibungen in der Datei `EXPERIMENTS.md`.
