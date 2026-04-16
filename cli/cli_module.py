from pathlib import Path
import subprocess
import sys
import os
from typing import List, Optional
import questionary
from questionary import Style
import colorama
import concurrent.futures

"""CLI wrapper and interactive runner for project experiments.

Contains `ClusterCLI` which provides a small TUI built on
`questionary` to list and run experiments and to inspect output files.
"""

from cli.cli_outputs import CLIOutputs

colorama.init()


class ClusterCLI:
    def __init__(self, root: Path):
        self.root: Path = root
        self.experiments: Path = self.root / "experiments"
        self.outputs = CLIOutputs(self.experiments)
        self.style = Style([
            ("pointer", "fg:ansigreen bold"),
            ("selected", "fg:ansigreen bold"),
            ("highlighted", "fg:ansigreen bold"),
            ("question", "bold"),
        ])

    def list_experiments(self) -> List[str]:
        if not self.experiments.exists():
            return []
        return sorted([p.name for p in self.experiments.iterdir() if p.is_dir()])

    def choose_experiment(self, tokens: List[str]) -> Optional[str]:
        if not tokens:
            return None        
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:
            pass
        return questionary.select("Select experiment:", choices=tokens, use_arrow_keys=True, style=self.style).ask()

    def run_experiment(self, token: str, show_header: bool = True) -> int:
        folder = self.experiments / token
        script = folder / f"{token}.py"
        cfg = folder / f"{token}.yaml"
        if not script.exists() or not cfg.exists():
            print(f"Missing script or config for {token}", file=sys.stderr)
            return 2
        cmd = [sys.executable, str(script), "--config", str(cfg)]
        if show_header:
            print()
            print(colorama.Fore.CYAN + f"=== {token} ===" + colorama.Style.RESET_ALL)
            print(colorama.Fore.YELLOW + "Running: " + " ".join(cmd) + colorama.Style.RESET_ALL)
            print()
        proc = subprocess.run(cmd)
        print()
        return proc.returncode

    def offer_open_outputs(self, token: str) -> None:
        files = self.outputs.outputs_for(token)
        if not files:
            print("No outputs found.")
            return
        htmls = [f for f in files if f.suffix.lower() in (".html", ".htm")]
        first_html = htmls[0] if htmls else None
        
        files_for_menu = [f for f in files if f != first_html] if first_html else files
        menu_choices: List[str] = []
        if first_html:
            menu_choices.append(f"Open HTML: {first_html.name}")
        menu_choices += [f.name for f in files_for_menu]
        menu_choices.append("Back")

        while True:
            choice = questionary.select("Open file (select 'Back' to return):", choices=menu_choices, use_arrow_keys=True, style=self.style).ask()
            if not choice or choice == "Back":
                break
            if first_html and choice.startswith("Open HTML:"):
                self.outputs.open_path(first_html)
                continue
            sel = next((f for f in files if f.name == choice), None)
            if sel:
                self.outputs.open_path(sel)
            else:
                print("Selection not found.", file=sys.stderr)

    def run(self) -> None:
        try:
            while True:
                tokens = self.list_experiments()
                if not tokens:
                    print(f"No experiments found in {self.experiments}", file=sys.stderr)
                    sys.exit(1)
                choices = ["Run all"] + tokens + ["Close"]
                chosen = self.choose_experiment(choices)
                if not chosen:
                    print("No selection, exiting.")
                    sys.exit(0)
                if chosen == "Close":
                    print("Closing...")
                    sys.exit(0)
                if chosen == "Run all":
                    cpu = os.cpu_count() or 1
                    # leave one core free by default, but at least 1 worker
                    max_workers = min(len(tokens), max(1, cpu - 1))
                    print(f"Running {len(tokens)} experiments in parallel ({max_workers} workers)...")
                    futures = {}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                        for t in tokens:
                            futures[ex.submit(self.run_experiment, t, False)] = t
                        results = {}
                        for fut in concurrent.futures.as_completed(futures):
                            t = futures[fut]
                            try:
                                rc = fut.result()
                            except Exception as exc:
                                rc = 1
                                print(f"Experiment {t} raised an exception: {exc}", file=sys.stderr)
                            results[t] = rc

                    print("\nRun all finished — summary:")
                    for t in tokens:
                        rc = results.get(t, 1)
                        status = "OK" if rc == 0 else f"EXIT {rc}"
                        print(f" - {t}: {status}")

                    # let user inspect outputs for any finished experiment
                    while True:
                        choices_status = [f"{t} ({'OK' if results.get(t,1)==0 else 'ERR'})" for t in tokens]
                        choices_status.append("Back")
                        pick = questionary.select("Inspect outputs for experiment:", choices=choices_status, use_arrow_keys=True, style=self.style).ask()
                        if not pick or pick == "Back":
                            break
                        picked_name = pick.split()[0]
                        self.offer_open_outputs(picked_name)
                    continue
                
                rc = self.run_experiment(chosen)
                if rc != 0:
                    print("Experiment exited with code", rc)
                self.offer_open_outputs(chosen)
        except KeyboardInterrupt:
            print("\nInterrupted, exiting.")
            sys.exit(0)
