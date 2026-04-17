"""Utilities for locating and opening experiment output files.

Provides a small class to list files under an experiment's `outputs`
directory and open them using the OS default application. HTML files
are opened via the `webbrowser` module to ensure they open in the
default browser.
"""

from pathlib import Path
import sys
import os
import subprocess
import webbrowser
from typing import List
from ruamel.yaml import YAML


class CLIOutputs:
    """Helper to list and open experiment output files.

    Initialized with the `experiments` root Path.
    Reads the output directory name from the experiment's YAML config.
    """

    def __init__(self, experiments_root: Path) -> None:
        self.experiments_root: Path = experiments_root
        self.yaml = YAML()

    def _get_output_dir_name(self, token: str) -> str:
        """Read output directory name from experiment config.
        
        Returns only the last segment of the output directory path
        (e.g., 'outputs2086' from 'experiments/affinityPropagation_tfidf/outputs2086').
        """
        cfg_path = self.experiments_root / token / f"{token}.yaml"
        if not cfg_path.exists():
            return "outputs"
        
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = self.yaml.load(f) or {}
            
            output_dir = data.get("outputs", {}).get("output_dir") or "outputs"
            return Path(output_dir).name
        except Exception:
            return "outputs"

    def outputs_for(self, token: str) -> List[Path]:
        output_dir_name = self._get_output_dir_name(token)
        outdir = self.experiments_root / token / output_dir_name
        if not outdir.exists():
            return []
        return sorted([p for p in outdir.iterdir() if p.is_file()])

    def open_path(self, p: Path) -> None:
        """Open a file using the platform default application.

        HTML files are opened in the default browser via `webbrowser`.
        Errors are caught and printed to stderr to avoid crashing the CLI.
        """
        if not p.exists():
            return
        try:
            suffix = p.suffix.lower()
            if suffix in (".html", ".htm"):
                webbrowser.open(p.as_uri())
                return
            if sys.platform.startswith("win"):
                os.startfile(str(p))
                return
            if sys.platform == "darwin":
                subprocess.run(["open", str(p)], check=False)
                return
            subprocess.run(["xdg-open", str(p)], check=False)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to open {p}: {exc}", file=sys.stderr)
