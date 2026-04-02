# spectral_paper_cluster

Simple starter setup for clustering experiments with PyTorch.

## Structure

- `configs/` runtime configs
- `configs/experiments/` optional experiment configs (if you parameterize runs)
- `data/raw/` raw datasets (kept with placeholder `.gitkeep`)
- `data/processed/` processed artifacts (ignored, with placeholder `.gitkeep`)
- `notebooks/` exploration notebooks
- `experiments/` runnable experiment pipelines (one file per experiment)
- `src/clustering/` clustering algorithms and models
- `src/features/` text feature extraction modules
- `src/pipelines/` end-to-end experiment orchestration
- `src/evaluation/` clustering metrics and reports
- `src/utils/` helper functions
- `outputs/` plots and run artifacts (ignored, with placeholder `.gitkeep`)

Result plot will be written to `outputs/clusters.png`.

## Run Pattern

- Put each executable experiment in `experiments/` (for example `experiments/tfidf_kmeans.py`).
- Keep reusable code in `src/` and call it from experiment files.
- Use `configs/experiments/` only when you want to externalize parameters.
