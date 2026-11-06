"""Convert the upstream cache without importing its API-calling Python modules."""
import csv
import hashlib
import io
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

ASPECTS = ('style', 'content', 'coordination', 'engagement')


class Record:
    pass


class CacheReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'obj.conversation' and name in {'Conversation', 'Aspect', 'Reason', 'Rebuttal'}:
            return Record
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name == 'scalar':
            return np._core.multiarray.scalar
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        raise pickle.UnpicklingError(f'Unsupported cache global: {module}.{name}')


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    temporary.replace(path)


def prepare(upstream, output):
    upstream, output = Path(upstream), Path(output)
    raw = (upstream / 'data/all_obj.pkl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'e4be3c7c6a3543a77ad07eddde44bfa42f6e58fa6a4bf47df53ea9e894f1e318':
        raise ValueError('Upstream cache checksum changed; review before importing')
    conversations = CacheReader(io.BytesIO(raw)).load()
    with (upstream / 'data/meeting.csv').open(newline='') as f:
        labels = list(csv.DictReader(f))
    if len(conversations) != 151 or len(labels) != 151:
        raise ValueError('Expected the released 151-snippet dataset')
    records = []
    for index, (conv, label) in enumerate(zip(conversations, labels)):
        if conv.text != label['conversation'] or conv.power != label['power']:
            raise ValueError(f'CSV/cache mismatch at row {index}')
        arguments, relations = [], []
        # Indexing is exactly the release's 98-variable graph: 2 power variables,
        # followed by four groups of 6 reasons, 6 rebuttals, 12 relations.
        for aspect_index, aspect in enumerate(ASPECTS):
            obj = getattr(conv, aspect)
            base = 2 + 24 * aspect_index
            for kind, offset in [('reason', 0), ('rebuttal', 6)]:
                for speaker_index, side in enumerate(('A', 'B')):
                    for j, arg in enumerate(getattr(obj, f'{kind}_{side.lower()}')):
                        variable = base + offset + 3 * speaker_index + j
                        arguments.append(dict(
                            id=f'{index}:{variable}', variable=variable, kind=kind,
                            # rebuttal_a is AGAINST reason_a, not by speaker A.
                            side=side, aspect=aspect, text=arg.text,
                            target=f'{index}:{base + 3 * speaker_index + j}' if kind == 'rebuttal' else None,
                            validity=float(arg.valid_potential),
                            grounding=float(getattr(arg, 'grounding_potential' if kind == 'reason' else 'rebuttal_potential'))))
            scores = obj.similarity_score + obj.contradict_score_a + obj.contradict_score_b
            for j, score in enumerate(scores):
                relations.append({'variable': base + 12 + j, 'score': int(score[0][0].strip()) / 5})
        if len(arguments) != 48 or len(relations) != 48:
            raise ValueError(f'Unexpected argument shape at row {index}')
        records.append(dict(id=str(index), transcript=conv.text, label=conv.power,
                            arguments=arguments, relations=relations))
    write(output / 'snippets.jsonl', records)
    rules = (upstream / 'rules/rules.txt').read_text()
    (output / 'rules.txt').write_text(rules)
    provenance = dict(source='https://github.com/kandicew/social-reasoning',
                      commit=subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                      pickle_sha256=hashlib.sha256(raw).hexdigest(), snippets=len(records),
                      reasons=sum(a['kind'] == 'reason' for r in records for a in r['arguments']),
                      rebuttals=sum(a['kind'] == 'rebuttal' for r in records for a in r['arguments']),
                      label_A=sum(r['label'] == 'A' for r in records))
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance
