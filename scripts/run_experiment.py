#!/usr/bin/env python3
"""Run the fixed experiment with resumable requests and explicit outcome files."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from quote_power.data import load
from quote_power.scoring import score

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
              planned_unique_requests=14647, prompt_control='same-model reason-only template',
              started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              plan='docs/experiment-plan.md')

def save_status():
    tmp = OUT / 'status.json.tmp'
    tmp.write_text(json.dumps(status, indent=2) + '\n')
    tmp.replace(OUT / 'status.json')

save_status()
try:
    rows = load(ROOT / 'data/processed/snippets.jsonl')
    if len(rows) != 151:
        raise ValueError('Experiment requires the complete released dataset')
    score(rows, status['model'], ROOT / 'results/experiment-cache', OUT)
    subprocess.run([sys.executable, '-m', 'quote_power.cli', 'evaluate', '--factorial',
                    '--annotations', str(OUT / 'annotations.jsonl'),
                    '--zero-shot', str(OUT / 'zero_shot.jsonl'), '--output', str(OUT / 'evaluation')], check=True)
    subprocess.run([sys.executable, '-m', 'quote_power.cli', 'audit',
                    '--annotations', str(OUT / 'annotations.jsonl'), '--output', str(OUT / 'audit.jsonl')], check=True)
    status['state'] = 'complete'
    status['human_audit'] = 'pending human judgments; worksheet exported'
except Exception as exc:
    status['state'] = 'blocked'
    status['error_type'] = type(exc).__name__
    # Only locally generated validation errors are recorded verbatim. SDK HTTP
    # error bodies/headers are deliberately excluded from published artifacts.
    if isinstance(exc, ValueError):
        status['validation_error'] = str(exc)
    raise
finally:
    status['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for field, filename in [('exported_arguments', 'annotations.jsonl'), ('exported_zero_shot', 'zero_shot.jsonl')]:
        path = OUT / filename
        status[field] = len(path.read_text().splitlines()) if path.exists() else 0
    save_status()
