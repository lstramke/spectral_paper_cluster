"""Small runner that delegates to the `cli` package.

This file remains in the project root and only bootstraps the new
`ClusterCLI` class implemented in `cli/cli_module.py`.
"""
from pathlib import Path

from cli.cli_module import ClusterCLI


if __name__ == "__main__":
    ClusterCLI(Path(__file__).resolve().parent).run()