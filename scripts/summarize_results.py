"""Recompute saved metrics and export factual tables, not report prose."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cli'))
from metrics import quality, leaderboard


def read_report(path):
    report = json.loads(path.read_text())
    rows = report['files']
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate IDs: {path}')
    q = quality(*[[r[k] for r in rows] for k in ('pesq', 'stoi', 'si_sdr')])
    for k, v in q.items():
        if not np.isclose(report[k], v, atol=1e-9, rtol=0):
            raise ValueError(f'Stored aggregate disagrees with file metrics: {path}: {k}')
    return q, {r['id']: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=Path('runs/baseline'))
    p.add_argument('--comparison', type=Path, help='Optional proposal evaluation directory')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows, maps = [], {}
    for label, filename in [('noisy_input', 'noisy_input_report.json'),
                             ('fullsubnet_baseline', 'noisy_report.json'),
                             ('clean_preservation', 'clean_preservation_report.json')]:
        q, by_id = read_report(a.run / filename)
        rows.append({'system': label, **q}); maps[label] = by_id
    if len({frozenset(v) for v in maps.values()}) != 1:
        raise ValueError('Report IDs differ')
    result = {'provenance': 'recomputed from supplied reports; audio/weights not re-evaluated',
              'baseline_local_score': leaderboard(rows[1]['quality'], rows[2]['quality']),
              'server_submission_verified': False}
    if a.comparison:
        q, proposal = read_report(a.comparison / 'noisy_report.json')
        cq, clean = read_report(a.comparison / 'clean_preservation_report.json')
        baseline = maps['fullsubnet_baseline']
        if set(proposal) != set(baseline) or set(clean) != set(baseline):
            raise ValueError('Comparison utterance IDs must match baseline')
        rows.extend([{'system': 'proposal', **q}, {'system': 'proposal_clean', **cq}])
        result['proposal_local_score'] = leaderboard(q['quality'], cq['quality'])
        rng = np.random.default_rng(523)
        intervals = {}
        for metric in ('pesq', 'stoi', 'si_sdr'):
            delta = np.array([proposal[i][metric]-baseline[i][metric] for i in sorted(baseline)])
            means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
            intervals[metric] = {'mean_difference': float(delta.mean()),
                                 'paired_utterance_95pct_interval': np.quantile(means, [.025, .975]).tolist()}
        result['proposal_minus_baseline'] = intervals
        result['interval_limit'] = 'Utterance resampling only; not uncertainty across seeds or speakers.'
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
