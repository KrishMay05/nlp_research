#!/usr/bin/env python3
"""Run the fixed experiment with resumable requests and explicit outcome files."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from quote_power.data import load
from quote_power.scoring import score

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--quote-only', action='store_true', help='Finish quote filtering with fixed cached scores; skip probability rescoring')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = ROOT / 'results/experiment'
OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT / 'run.lock').open('w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('An experiment runner is already active')

status = dict(state='running', model='gemma4:26b-a4b', target_snippets=151,
              planned_unique_requests=7399 if args.quote_only else 14647,
              scope='quote_filter_only' if args.quote_only else 'full_factorial',
              prompt_control='fixed released cached weights' if args.quote_only else 'same-model reason-only template',
              started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              plan='docs/experiment-plan.md')

def save_status():
    tmp = OUT / 'status.json.tmp'
    tmp.write_text(json.dumps(status, indent=2) + '\n')
    tmp.replace(OUT / 'status.json')

previous_path = OUT / 'status.json'
if previous_path.exists():
    history_path = OUT / 'attempt_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(json.loads(previous_path.read_text()))
    history_path.write_text(json.dumps(history, indent=2) + '\n')
save_status()
try:
    rows = load(ROOT / 'data/processed/snippets.jsonl')
    if len(rows) != 151:
        raise ValueError('Experiment requires the complete released dataset')
    annotation_file = 'quote_annotations.jsonl' if args.quote_only else 'annotations.jsonl'
    evaluation_dir = 'quote_evaluation' if args.quote_only else 'evaluation'
    score(rows, status['model'], ROOT / 'results/experiment-cache', OUT, quote_only=args.quote_only)
    subprocess.run([sys.executable, '-m', 'quote_power.cli', 'evaluate', '--quote-only' if args.quote_only else '--factorial',
                    '--annotations', str(OUT / annotation_file),
                    '--zero-shot', str(OUT / 'zero_shot.jsonl'), '--output', str(OUT / evaluation_dir)], check=True)
    subprocess.run([sys.executable, '-m', 'quote_power.cli', 'audit',
                    '--annotations', str(OUT / annotation_file), '--output', str(OUT / 'audit.jsonl')], check=True)
    status['state'] = 'complete'
    status['human_audit'] = 'pending human judgments; worksheet exported'
except Exception as exc:
    status['state'] = 'blocked'
    status['error_type'] = type(exc).__name__
    body = getattr(exc, 'body', None)
    if isinstance(body, dict) and body.get('detail') == 'Open WebUI: Server Connection Error':
        status['service_error'] = 'Purdue Open WebUI backend connection failure'
    # Only locally generated validation errors are recorded verbatim. SDK HTTP
    # error bodies/headers are deliberately excluded from published artifacts.
    if isinstance(exc, ValueError):
        status['validation_error'] = str(exc)
    raise
finally:
    status['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for field, filename in [('exported_arguments', 'quote_annotations.jsonl' if args.quote_only else 'annotations.jsonl'), ('exported_zero_shot', 'zero_shot.jsonl')]:
        path = OUT / filename
        status[field] = len(path.read_text().splitlines()) if path.exists() else 0
    save_status()
