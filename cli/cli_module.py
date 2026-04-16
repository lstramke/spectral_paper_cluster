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
from cli.cli_config_editor import CLIConfigEditor

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
        self.config_editor = CLIConfigEditor(style=self.style)

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
            print(colorama.Style.BRIGHT + colorama.Fore.RED + f"Missing script or config for {token}" + colorama.Style.RESET_ALL, file=sys.stderr)
            return 2
        cmd = [sys.executable, str(script), "--config", str(cfg)]
        if show_header:
            print()
            print(colorama.Style.BRIGHT + colorama.Fore.CYAN + f"  ▶ {token}" + colorama.Style.RESET_ALL)
        proc = subprocess.run(cmd)
        if show_header:
            print()
        return proc.returncode

    def offer_open_outputs(self, token: str) -> None:
        files = self.outputs.outputs_for(token)
        if not files:
            print(colorama.Fore.YELLOW + "⚠ No outputs found." + colorama.Style.RESET_ALL)
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
                print(colorama.Style.BRIGHT + colorama.Fore.RED + "Selection not found." + colorama.Style.RESET_ALL, file=sys.stderr)

    def edit_config_menu(self, tokens: List[str]) -> None:
        """Show the edit-config submenu and open the config editor."""
        pick = questionary.select("Select experiment to edit config:", choices=tokens + ["Back"], use_arrow_keys=True, style=self.style).ask()
        if not pick or pick == "Back":
            return
        cfg = self.experiments / pick / f"{pick}.yaml"
        self.config_editor.edit_config(cfg)

    def experiments_menu(self, tokens: List[str]) -> None:
        """Show the experiments submenu (run all, run one, inspect outputs)."""
        choices = ["Run all"] + tokens + ["Back"]
        chosen = self.choose_experiment(choices)
        if not chosen or chosen == "Back":
            return
        if chosen == "Run all":
            results = self.run_all(tokens)
            ok_count = sum(1 for rc in results.values() if rc == 0)
            total_count = len(results)
            status_color = colorama.Fore.GREEN if ok_count == total_count else colorama.Fore.YELLOW if ok_count > 0 else colorama.Fore.RED
            print()
            print(colorama.Style.DIM + colorama.Fore.WHITE + "═" * 60 + colorama.Style.RESET_ALL)
            print(status_color + "✓" + colorama.Style.RESET_ALL + colorama.Style.BRIGHT + f" Run all finished: {ok_count}/{total_count} successful" + colorama.Style.RESET_ALL)
            print(colorama.Style.DIM + colorama.Fore.WHITE + "═" * 60 + colorama.Style.RESET_ALL)
            print()

            while True:
                choices_status = [f"{t} ({'OK' if results.get(t,1)==0 else 'ERR'})" for t in tokens]
                choices_status.append("Back")
                pick = questionary.select("Inspect outputs for experiment:", choices=choices_status, use_arrow_keys=True, style=self.style).ask()
                if not pick or pick == "Back":
                    break
                picked_name = pick.split()[0]
                self.offer_open_outputs(picked_name)
            return

        rc = self.run_experiment(chosen)
        if rc != 0:
            print(colorama.Fore.RED + "✗" + colorama.Style.RESET_ALL + colorama.Style.BRIGHT + f" Experiment exited with code {rc}" + colorama.Style.RESET_ALL)
        else:
            print(colorama.Fore.GREEN + "✓" + colorama.Style.RESET_ALL + colorama.Style.BRIGHT + " Experiment completed successfully" + colorama.Style.RESET_ALL)
        print()
        self.offer_open_outputs(chosen)

    def start(self) -> None:
        try:
            while True:
                tokens = self.list_experiments()
                if not tokens:
                    print(colorama.Style.BRIGHT + colorama.Fore.RED + f"No experiments found in {self.experiments}" + colorama.Style.RESET_ALL, file=sys.stderr)
                    sys.exit(1)
                main_choices = ["Edit config", "Experiments", "Close"]
                action = questionary.select("Select action:", choices=main_choices, use_arrow_keys=True, style=self.style).ask()
                if not action:
                    print(colorama.Style.BRIGHT + colorama.Fore.YELLOW + "No selection, exiting." + colorama.Style.RESET_ALL)
                    sys.exit(0)
                if action == "Close":
                    print(colorama.Style.BRIGHT + " Closing..." + colorama.Style.RESET_ALL)
                    sys.exit(0)
                if action == "Edit config":
                    self.edit_config_menu(tokens)
                    continue
                self.experiments_menu(tokens)
        except KeyboardInterrupt:
            print(colorama.Style.BRIGHT + colorama.Fore.YELLOW + "\n⚠ Interrupted, exiting." + colorama.Style.RESET_ALL)
            sys.exit(0)

    def run_all(self, tokens):
        cpu = os.cpu_count() or 1
        # leave one core free by default, but at least 1 worker
        max_workers = min(len(tokens), max(1, cpu - 1))
        print()
        print(colorama.Style.BRIGHT + colorama.Fore.CYAN + f"▶ Running {len(tokens)} experiments in parallel ({max_workers} workers)..." + colorama.Style.RESET_ALL)
        print()
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
                symbol = "✓" if rc == 0 else "✗"
                status = f"EXIT {rc}" if rc != 0 else ""
                color = colorama.Fore.GREEN if rc == 0 else colorama.Fore.RED
                msg = f" {t}" if not status else f" {t}: {status}"
                print(color + symbol + colorama.Style.RESET_ALL + colorama.Style.BRIGHT + msg + colorama.Style.RESET_ALL, flush=True)
        return results
