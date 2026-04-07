from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import yaml

from src.clustering.kmeans import KMeansConfig
from src.features.tfidf import TfidfConfig
from src.interpretation.tfidf_interpreter import TfidfInterpreterConfig


@dataclass(slots=True)
class InputConfig:
    documents_path: Path
    format: str
    text_fields: list[str]
    fuse_mode: str
    separator: str


@dataclass(slots=True)
class OutputsConfig:
    output_dir: Path
    plot_name: str
    summary_name: str
    point_size: int
    alpha: float
    figsize_width: float
    figsize_height: float


@dataclass(slots=True)
class ParsedExperimentConfig:
    experiment_name: str
    input: InputConfig
    kmeans: KMeansConfig
    tfidf: TfidfConfig
    interpretation: TfidfInterpreterConfig
    outputs: OutputsConfig


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if raw is None:
        raise ValueError("Config is empty")
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return cast(dict[str, Any], raw)


def require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping: '{key}'")
    return cast(dict[str, Any], value)


def require_value(parent: dict[str, Any], key: str) -> Any:
    if key not in parent:
        raise ValueError(f"Missing required config key: '{key}'")
    return parent[key]


def validate_and_parse_config(config: dict[str, Any], project_root: Path) -> ParsedExperimentConfig:
    experiment_name = str(require_value(config, "experiment_name"))

    input_cfg = require_mapping(config, "input")
    documents_path = project_root / str(require_value(input_cfg, "documents_path"))
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents file not found: {documents_path}")

    input_format = str(require_value(input_cfg, "format"))
    if input_format not in {"line", "csv"}:
        raise ValueError("input.format must be either 'line' or 'csv'")

    text_fields: list[str] = []
    fuse_mode = ""
    separator = ""
    if input_format == "csv":
        text_fields_raw = require_value(input_cfg, "text_fields")
        if not isinstance(text_fields_raw, list) or not text_fields_raw:
            raise ValueError("input.text_fields must be a non-empty list for csv format")
        text_fields_seq = cast(Sequence[Any], text_fields_raw)
        text_fields = [str(field) for field in text_fields_seq]

        fuse_mode = str(require_value(input_cfg, "fuse_mode"))
        if fuse_mode not in {"join", "first_non_empty"}:
            raise ValueError("input.fuse_mode must be one of: join, first_non_empty")
        separator = str(require_value(input_cfg, "separator"))

    pipeline_cfg = require_mapping(config, "pipeline")
    pipeline_n_clusters = int(require_value(pipeline_cfg, "n_clusters"))
    pipeline_max_iter = int(require_value(pipeline_cfg, "max_iter"))
    pipeline_tol = float(require_value(pipeline_cfg, "tol"))
    pipeline_seed_range: tuple[int, int] | None = None
    if "seed_range" in pipeline_cfg:
        seed_range_raw: Any = require_value(pipeline_cfg, "seed_range")
        if not isinstance(seed_range_raw, (list, tuple)):
            raise ValueError("pipeline.seed_range must be a list or tuple with two integers")
        seed_range_values = cast(Sequence[Any], seed_range_raw)
        if len(seed_range_values) != 2:
            raise ValueError("pipeline.seed_range must have exactly two values")
        seed_start = int(seed_range_values[0])
        seed_end = int(seed_range_values[1])
        if seed_start > seed_end:
            raise ValueError("pipeline.seed_range start must be <= end")
        pipeline_seed_range = (seed_start, seed_end)
        pipeline_seed = seed_start
    else:
        pipeline_seed = int(require_value(pipeline_cfg, "seed"))

    kmeans = KMeansConfig(
        n_clusters=pipeline_n_clusters,
        max_iter=pipeline_max_iter,
        tol=pipeline_tol,
        seed=pipeline_seed,
        seed_range=pipeline_seed_range,
    )

    tfidf_cfg = require_mapping(config, "tfidf")
    ngram_range_raw: Any = require_value(tfidf_cfg, "ngram_range")
    if not isinstance(ngram_range_raw, (list, tuple)):
        raise ValueError("tfidf.ngram_range must be a list with two integers")
    ngram_values = cast(Sequence[Any], ngram_range_raw)
    if len(ngram_values) != 2:
        raise ValueError("tfidf.ngram_range must have exactly two values")

    use_lsa = bool(require_value(tfidf_cfg, "use_lsa"))
    lsa_components = int(require_value(tfidf_cfg, "lsa_components"))
    if lsa_components < 2:
        raise ValueError("tfidf.lsa_components must be >= 2")

    tfidf = TfidfConfig(
        max_features=int(require_value(tfidf_cfg, "max_features")),
        ngram_range=(int(ngram_values[0]), int(ngram_values[1])),
        min_df=cast(int | float, require_value(tfidf_cfg, "min_df")),
        max_df=cast(int | float, require_value(tfidf_cfg, "max_df")),
        lowercase=bool(require_value(tfidf_cfg, "lowercase")),
        stop_words=cast(str | list[str] | None, require_value(tfidf_cfg, "stop_words")),
        use_lsa=use_lsa,
        lsa_components=lsa_components,
    )

    interpretation_cfg = require_mapping(config, "interpretation")
    top_n_terms = int(require_value(interpretation_cfg, "top_n_terms"))
    if top_n_terms < 1:
        raise ValueError("interpretation.top_n_terms must be >= 1")
    interpretation = TfidfInterpreterConfig(top_n_terms=top_n_terms)

    outputs_cfg = require_mapping(config, "outputs")
    output_dir = project_root / str(require_value(outputs_cfg, "output_dir"))
    plot_name = str(require_value(outputs_cfg, "plot_name"))
    summary_name = str(require_value(outputs_cfg, "summary_name"))
    point_size = int(require_value(outputs_cfg, "point_size"))
    alpha = float(require_value(outputs_cfg, "alpha"))
    figsize_width = float(require_value(outputs_cfg, "figsize_width"))
    figsize_height = float(require_value(outputs_cfg, "figsize_height"))

    if point_size <= 0:
        raise ValueError("outputs.point_size must be > 0")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("outputs.alpha must be in (0, 1]")
    if figsize_width <= 0 or figsize_height <= 0:
        raise ValueError("outputs.figsize_width and figsize_height must be > 0")

    return ParsedExperimentConfig(
        experiment_name=experiment_name,
        input=InputConfig(
            documents_path=documents_path,
            format=input_format,
            text_fields=text_fields,
            fuse_mode=fuse_mode,
            separator=separator,
        ),
        kmeans=kmeans,
        tfidf=tfidf,
        interpretation=interpretation,
        outputs=OutputsConfig(
            output_dir=output_dir,
            plot_name=plot_name,
            summary_name=summary_name,
            point_size=point_size,
            alpha=alpha,
            figsize_width=figsize_width,
            figsize_height=figsize_height,
        ),
    )
