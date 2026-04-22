from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_summary(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def extract_row(exp_folder: str, result_dir: Path, data: Dict) -> Dict[str, object]:
    j = data
    metrics = j.get('metrics', {}) if isinstance(j, dict) else {}
    silhouette = metrics.get('silhouette', '')
    davies = metrics.get('davies_bouldin', '')
    calinski = metrics.get('calinski_harabasz', '')

    n_clusters = j.get('n_clusters_found') if isinstance(j, dict) else None
    # some summaries use 'n_clusters_found', some 'n_clusters' or 'n_clusters_found'
    if n_clusters is None:
        n_clusters = j.get('n_clusters') if isinstance(j, dict) else None
    if n_clusters is None:
        n_clusters = ''

    return {
        'experiment': exp_folder,
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'davies_bouldin': davies,
        'calinski_harabasz': calinski,
    }

def build_table(dataset: str, rows: List[Dict]) -> str:
    lines: List[str] = []
    lines.append(f'## Dataset: {dataset}\n')
    lines.append('| Experiment | n_clusters | silhouette | davies_bouldin | calinski_harabasz | summary markdown |')
    lines.append('| --- | ---: | ---: | ---: | ---: | --- |')

    suffix = dataset.split('_')[-1]
    for r in sorted(rows, key=lambda x: x['experiment']):
        exp = r['experiment']
        name_display = f"{exp}"
        md_path = f"experiments/{exp}/results_{suffix}/{exp}_{suffix}.md"
        lines.append(f"| {name_display} | {r['n_clusters']} | {r['silhouette']} | {r['davies_bouldin']} | {r['calinski_harabasz']} | [summary]({md_path}) |")

    lines.append('\n')
    return '\n'.join(lines)

def main() -> None:
    ap = argparse.ArgumentParser(description='Generate comparison tables from experiment summary JSONs')
    ap.add_argument('--experiments-dir', default='experiments')
    ap.add_argument('--output', default='SUMMARY.md')
    args = ap.parse_args()

    exp_root = Path(args.experiments_dir)
    json_paths = list(exp_root.glob('*/results_*/best_*_summary.json'))

    by_dataset: Dict[str, List[Dict]] = {}

    for p in json_paths:
        j = load_summary(p)
        if j is None:
            continue
        result_dir = p.parent.name
        exp_folder = p.parent.parent.name
        row = extract_row(exp_folder, p.parent, j)
        by_dataset.setdefault(result_dir, []).append(row)

    datasets = [d for d in ['results_41', 'results_2086'] if d in by_dataset]

    out_lines: List[str] = []
    out_lines.append('# Experimentzusammenfassung\n')
    out_lines.append('Tabelle zum Vergleich der Experimente nach Cluster-Anzahl und Metriken. Jede Zeile verlinkt auf die jeweilige Summary-Datei im `experiments/`-Ordner.\n')

    for ds in datasets:
        rows = by_dataset.get(ds, [])
        out_lines.append(build_table(ds, rows))

    Path(args.output).write_text('\n'.join(out_lines), encoding='utf-8')
    print(f'Wrote {args.output} with datasets: {", ".join(datasets)}')


if __name__ == '__main__':
    main()
