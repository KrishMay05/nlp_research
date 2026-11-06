import math
import re
from collections import defaultdict

import pulp


def normalize(text):
    return ' '.join(text.lower().split())


def quote_matches(transcript, quote, speaker=None):
    if not isinstance(quote, str) or not normalize(quote) or normalize(quote) == 'none':
        return False
    if speaker is None:
        return normalize(quote) in normalize(transcript)
    # Match within a complete speaker turn; never across two speakers' turns.
    turns = re.split(r'(?m)^\s*([AB]):', transcript)
    return any(turns[i] == speaker and normalize(quote) in normalize(turns[i + 1])
               for i in range(1, len(turns) - 1, 2))


def kept(row, annotations, quote_filter):
    if not quote_filter:
        return row['arguments']
    result = []
    for arg in row['arguments']:
        ann = annotations.get(arg['id'])
        if ann is None:
            raise ValueError(f"Missing quote annotation for {arg['id']}")
        # Evidence speaker is explicit, since a rebuttal may cite either speaker.
        speaker = ann.get('evidence_speaker')
        if speaker in {'A', 'B'} and ann.get('supports') is True and ann.get('attribution_correct') is True and quote_matches(row['transcript'], ann.get('quote'), speaker):
            result.append(arg)
    return result


def grounded_labels(nodes, attacks):
    nodes = set(nodes)
    attackers = defaultdict(set)
    for source, target in attacks:
        if source not in nodes or target not in nodes:
            raise ValueError('Attack endpoint missing')
        attackers[target].add(source)
    inside, outside = set(), set()
    while True:
        new_in = {n for n in nodes - inside - outside if attackers[n] <= outside}
        new_out = {n for n in nodes - inside - outside if attackers[n] & (inside | new_in)}
        if not new_in and not new_out:
            break
        inside |= new_in
        outside |= new_out
    return {n: 'IN' if n in inside else 'OUT' if n in outside else 'UNDECIDED' for n in nodes}


def grounding(arg, annotations, corrected):
    field = 'rebuttal_probability' if corrected else 'original_rebuttal_probability'
    if arg['kind'] == 'rebuttal' and (corrected or field in annotations.get(arg['id'], {})):
        probability = annotations.get(arg['id'], {}).get(field)
        if probability is None or not 0 <= probability <= 1:
            raise ValueError(f"Missing/invalid corrected score for {arg['id']}")
        return math.log(max(probability, 1e-12))
    return arg['grounding']


def grounded(row, annotations=None, quote_filter=False, corrected=False):
    annotations = annotations or {}
    args = kept(row, annotations, quote_filter)
    reasons = [a for a in args if a['kind'] == 'reason']
    ids = {a['id'] for a in args}
    attacks = [(a['id'], a['target']) for a in args if a['kind'] == 'rebuttal' and a['target'] in ids]
    attacks += [(a['id'], b['id']) for a in reasons for b in reasons if a['side'] != b['side']]
    labels = grounded_labels(ids, attacks)
    counts = {s: sum(labels[a['id']] == 'IN' for a in reasons if a['side'] == s) for s in ('A', 'B')}
    sums = {s: sum(math.exp(a['grounding']) for a in reasons if a['side'] == s) for s in ('A', 'B')}
    # Corrected rebuttal scores do not threshold attacks: every retained rebuttal
    # attacks, per the proposal. The corrected grounded arms are thus identical.
    prediction = max(('A', 'B'), key=lambda s: (counts[s], sums[s], s == 'A'))
    route = ('in_reason_count' if counts['A'] != counts['B'] else
             'grounding_score_tie_break' if sums['A'] != sums['B'] else 'final_A_tie_break')
    return dict(prediction=prediction, retained=len(args), labels=labels,
                accepted_reason_count=sum(counts.values()), kept_reason_count=len(reasons),
                decision_route=route)


def xor(row, rules, annotations=None, quote_filter=False, corrected=False, relaxation=False):
    annotations = annotations or {}
    args = kept(row, annotations, quote_filter)
    if not any(a['kind'] == 'reason' for a in args):
        return dict(prediction='A', retained=len(args), power_A=1., power_B=0.,
                    objective=None, fallback='no_surviving_reasons')
    active = {a['variable'] for a in args}
    model = pulp.LpProblem('power', pulp.LpMaximize)
    x = [pulp.LpVariable(f'x{i}', 0, 1, cat='Continuous' if relaxation else 'Binary') for i in range(98)]
    weights = {}
    for arg in args:
        weight = 20 if arg['kind'] == 'reason' else 18
        # Match the release's exp(sum(log potentials)), rounded to 2 decimals.
        weights[arg['variable']] = weight * round(math.exp(arg['validity'] + grounding(arg, annotations, corrected)), 2)
    for rel in row['relations']:
        weights[rel['variable']] = rel['score']
    model += pulp.lpSum(w * x[i] for i, w in weights.items())
    for arg in row['arguments']:
        if arg['variable'] not in active:
            model += x[arg['variable']] == 0
    for line in rules.splitlines():
        if not line.strip():
            continue
        kind, count, *tokens = line.split()
        variables = [int(t) for t in tokens]
        if int(count) != len(variables):
            raise ValueError('Invalid factor arity')
        # Filtering removes a reason/rebuttal XOR factor when either endpoint is
        # removed, rather than forcing its surviving partner to be true.
        if kind == 'XOR' and variables != [1, 2] and any(abs(i) - 1 not in active for i in variables):
            continue
        literals = [x[i - 1] if i > 0 else 1 - x[-i - 1] for i in variables]
        if kind == 'XOR':
            model += pulp.lpSum(literals) == 1
        elif kind == 'OR':
            model += pulp.lpSum(literals) >= 1
        elif kind == 'OROUT':
            *inputs, output = literals
            for v in inputs:
                model += v <= output
            model += output <= pulp.lpSum(inputs)
        else:
            raise ValueError(f'Unknown factor {kind}')
    solver = pulp.PULP_CBC_CMD(msg=False, threads=1)
    if model.solve(solver) != pulp.LpStatusOptimal:
        raise RuntimeError(f'CBC failed: {pulp.LpStatus[model.status]}')
    a, b = pulp.value(x[0]), pulp.value(x[1])
    return dict(prediction='A' if a >= b else 'B', retained=len(args), power_A=a, power_B=b,
                objective=pulp.value(model.objective))
