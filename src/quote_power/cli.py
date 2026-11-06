import argparse
import json
import random
from pathlib import Path

from .data import load, prepare, write
from .evaluation import accuracy, paired
from .inference import grounded, xor


def keyed(path):
    records = load(path)
    result = {r['id']: r for r in records}
    if len(result) != len(records):
        raise ValueError(f'Duplicate IDs in {path}')
    return result


def evaluate(args):
    rows = load(args.data)
    gold = [r['label'] for r in rows]
    rules = Path(args.rules).read_text()
    annotations = keyed(args.annotations) if args.annotations else {}
    if args.factorial and not args.annotations:
        raise ValueError('--factorial requires complete --annotations')
    methods = {'majority_A': [dict(prediction='A', retained=48) for r in rows]}
    methods['xor_cached'] = [xor(r, rules) for r in rows]
    methods['grounded_cached'] = [grounded(r) for r in rows]
    if args.factorial:
        required = {a['id'] for r in rows for a in r['arguments']}
        if required != set(annotations):
            raise ValueError('Annotations must cover exactly all dataset arguments')
        for q in (False, True):
            for c in (False, True):
                for agg in ('xor', 'grounded'):
                    # Validate all corrected scores, even where Dung ignores weights.
                    if c:
                        from .inference import grounding
                        for row in rows:
                            for arg in row['arguments']:
                                grounding(arg, annotations, True)
                    methods[f'{agg}_quote{int(q)}_corrected{int(c)}'] = [
                        xor(r, rules, annotations, q, c) if agg == 'xor' else grounded(r, annotations, q, c) for r in rows]
    if args.zero_shot:
        zero = keyed(args.zero_shot)
        if set(zero) != {r['id'] for r in rows}:
            raise ValueError('Zero-shot predictions must cover exactly all snippets')
        methods['zero_shot'] = [zero[r['id']] for r in rows]
    predictions = {k: [r['prediction'] for r in values] for k, values in methods.items()}
    report = dict(snippets=len(rows), published_xor_accuracy=.543, methods={})
    for name, values in methods.items():
        report['methods'][name] = dict(accuracy=accuracy(gold, predictions[name]),
            argument_coverage=(sum(r.get('retained', 0) for r in values) / sum(len(r['arguments']) for r in rows)) if name not in {'zero_shot', 'majority_A'} else None,
            comparisons={b: paired(gold, predictions[name], predictions[b]) for b in ('majority_A', 'xor_cached', 'zero_shot') if b in predictions and b != name})
    report['xor_published_delta'] = report['methods']['xor_cached']['accuracy'] - .543
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    write(out / 'predictions.jsonl', [dict(id=r['id'], label=r['label'], predictions={m: predictions[m][i] for m in methods}) for i, r in enumerate(rows)])
    print(json.dumps({name: value['accuracy'] for name, value in report['methods'].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Quote-grounded conversational power experiments')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--upstream', required=True)
    p.add_argument('--output', default='data/processed')
    p = commands.add_parser('evaluate')
    p.add_argument('--data', default='data/processed/snippets.jsonl')
    p.add_argument('--rules', default='data/processed/rules.txt')
    p.add_argument('--output', default='results/offline')
    p.add_argument('--annotations')
    p.add_argument('--zero-shot')
    p.add_argument('--factorial', action='store_true')
    p = commands.add_parser('score')
    p.add_argument('--data', default='data/processed/snippets.jsonl')
    p.add_argument('--model', required=True)
    p.add_argument('--base-url')
    p.add_argument('--cache', default='results/api-cache')
    p.add_argument('--output', default='results/scored')
    p.add_argument('--limit', type=int, help='Smoke run only; cannot satisfy full evaluation')
    p = commands.add_parser('audit')
    p.add_argument('--data', default='data/processed/snippets.jsonl')
    p.add_argument('--annotations')
    p.add_argument('--output', default='results/audit.jsonl')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--count', type=int, default=40)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.upstream, args.output), indent=2))
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'score':
        from .scoring import score
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
        score(load(args.data), args.model, args.cache, args.output, args.base_url, args.limit)
    else:
        annotations = keyed(args.annotations) if args.annotations else {}
        pool = [dict(snippet_id=r['id'], transcript=r['transcript'], **a,
                     annotation=annotations.get(a['id']),
                     paired_reason=next((b['text'] for b in r['arguments'] if b['id'] == a['target']), None),
                     human_quote_matches=None, human_supports=None, human_answers_reason=None, notes='')
                for r in load(args.data) for a in r['arguments']]
        write(args.output, random.Random(args.seed).sample(pool, args.count))


if __name__ == '__main__':
    main()
