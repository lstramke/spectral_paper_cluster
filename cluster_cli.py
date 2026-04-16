
"""
Clusterer CLI — simple TUI to pick and run experiments with arrow keys.
Place this file in the project root and run `python cluster_cli.py`.
"""
from pathlib import Path
import subprocess
import sys

try:
    import questionary
    from questionary import Style
    import colorama
    colorama.init()
except Exception:
    print("Missing dependency: install with `pip install questionary colorama'", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"

def list_experiments():
    if not EXPERIMENTS.exists():
        return []
    return sorted([p.name for p in EXPERIMENTS.iterdir() if p.is_dir()])

def choose_experiment(tokens):
    if not tokens:
        return None
    style = Style([
        ("pointer", "fg:ansigreen bold"),
        ("selected", "fg:ansigreen bold"),
        ("highlighted", "fg:ansigreen bold"),
        ("question", "bold"),
    ])
    choice = questionary.select("Select experiment:", choices=tokens, use_arrow_keys=True, style=style).ask()
    return choice

def run_experiment(token: str) -> int:
    folder = EXPERIMENTS / token
    script = folder / f"{token}.py"
    cfg = folder / f"{token}.yaml"
    if not script.exists() or not cfg.exists():
        print(f"Missing script or config for {token}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script), "--config", str(cfg)]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode

def main():
    try:
        while True:
            tokens = list_experiments()
            if not tokens:
                print(f"No experiments found in {EXPERIMENTS}", file=sys.stderr)
                sys.exit(1)
            choices = ["Close"] + tokens
            chosen = choose_experiment(choices)
            if not chosen:
                print("No selection, exiting.")
                sys.exit(0)
            if chosen == "Close":
                print("Closing...")
                sys.exit(0)
            rc = run_experiment(chosen)
            if rc != 0:
                print("Experiment exited with code", rc)
    except KeyboardInterrupt:
        print("\nInterrupted, exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()