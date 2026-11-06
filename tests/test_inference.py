import math
from pathlib import Path

import pytest

from quote_power.data import load
from quote_power.evaluation import accuracy, paired
from quote_power.inference import grounded, grounded_labels, quote_matches, xor
from quote_power.scoring import quote_prompt, rebuttal_prompt


def test_grounded_chain_and_cycle():
    assert grounded_labels({'a', 'b', 'c'}, [('a', 'b'), ('b', 'c')]) == {'a': 'IN', 'b': 'OUT', 'c': 'IN'}
    assert set(grounded_labels({'a', 'b'}, [('a', 'b'), ('b', 'a')]).values()) == {'UNDECIDED'}
    with pytest.raises(ValueError):
        grounded_labels({'a'}, [('b', 'a')])


def test_quotes_are_normalized_and_speaker_bound():
    transcript = 'A: Set   the agenda.\nB: I agree.'
    assert quote_matches(transcript, 'SET the agenda.', 'A')
    assert not quote_matches(transcript, 'Set the agenda.', 'B')
    assert not quote_matches(transcript, 'NONE', 'A')
    assert not quote_matches(transcript, '', 'A')
    assert not quote_matches(transcript, 'agenda. B: I agree.', 'A')


def fixture_row():
    return dict(id='0', transcript='A: Set the agenda.\nB: I agree.', label='A', relations=[], arguments=[
        dict(id='a', variable=2, side='A', kind='reason', text='A directs.', validity=0, grounding=math.log(.9), target=None),
        dict(id='b', variable=5, side='B', kind='reason', text='B directs.', validity=0, grounding=math.log(.4), target=None),
        dict(id='r', variable=8, side='A', kind='rebuttal', text='A only suggests.', validity=0, grounding=0, target='a')])


def test_rebuttal_attack_and_fixed_tie_rules():
    row = fixture_row()
    result = grounded(row)
    assert result['labels'] == {'a': 'OUT', 'r': 'IN', 'b': 'IN'}
    assert result['prediction'] == 'B'
    row['arguments'] = row['arguments'][:2]
    assert grounded(row)['prediction'] == 'A'  # unresolved cycle, grounding tie break
    row['arguments'][1]['grounding'] = math.log(.9)
    assert grounded(row)['prediction'] == 'A'  # final tie


def test_filter_keeps_snippet_and_requires_semantic_checks():
    row = fixture_row()
    anns = {a['id']: dict(quote='Set the agenda.', evidence_speaker='B', supports=True, attribution_correct=True) for a in row['arguments']}
    assert grounded(row, anns, True)['retained'] == 0
    assert grounded(row, anns, True)['prediction'] == 'A'
    anns['a']['evidence_speaker'] = 'A'
    assert grounded(row, anns, True)['retained'] == 1
    anns['a']['supports'] = False
    assert grounded(row, anns, True)['retained'] == 0
    with pytest.raises(ValueError, match='Missing quote'):
        grounded(row, {}, True)


def test_xor_corrected_probability_changes_winner():
    row = fixture_row()
    # Power == its sole reason; A reason XOR rebuttal. B can win only
    # when the rebuttal is preferable to the reason.
    rules = 'XOR 2 1 2\nOROUT 2 3 1\nOROUT 2 6 2\nXOR 2 3 9'
    assert xor(row, rules)['prediction'] == 'B'
    assert xor(row, rules, {'r': {'rebuttal_probability': .01}}, corrected=True)['prediction'] == 'A'
    with pytest.raises(ValueError, match='corrected score'):
        xor(row, rules, corrected=True)


def test_statistics_pairing_and_full_coverage():
    result = paired(['A', 'B'], ['A', 'B'], ['B', 'A'])
    assert result['accuracy_delta'] == 1
    assert result['ci95'] == [1, 1]
    assert result['mcnemar_exact_p'] == .5
    assert paired(['A'], ['A'], ['A'])['mcnemar_exact_p'] == 1
    with pytest.raises(ValueError):
        accuracy(['A', 'B'], ['A'])


def test_prompts_name_rebuttal_and_reason():
    row = fixture_row()
    prompt = rebuttal_prompt(row, row['arguments'][2])
    assert 'A only suggests.' in prompt and 'A directs.' in prompt
    assert 'attribution_correct' in quote_prompt(row, row['arguments'][2])


def test_released_data_and_reproduction():
    root = Path(__file__).parents[1]
    path = root / 'data/processed/snippets.jsonl'
    if not path.exists():
        pytest.skip('Run scripts/setup.sh to acquire released data')
    rows = load(path)
    rules = (path.parent / 'rules.txt').read_text()
    assert len(rows) == 151
    assert sum(r['label'] == 'A' for r in rows) == 83
    assert sum(len(r['arguments']) for r in rows) == 7248
    results = [xor(r, rules)['prediction'] for r in rows]
    assert sum(r['label'] == p for r, p in zip(rows, results)) == 82


def test_xor_empty_evidence_uses_declared_tie_rule():
    row = fixture_row()
    anns = {a['id']: dict(quote='NONE', evidence_speaker='A', supports=False, attribution_correct=False) for a in row['arguments']}
    rules = 'XOR 2 1 2\nOROUT 2 3 1\nOROUT 2 6 2\nXOR 2 3 9'
    result = xor(row, rules, anns, quote_filter=True)
    assert result['prediction'] == 'A'
    assert result['retained'] == 0
    assert result['fallback'] == 'no_surviving_reasons'
    assert result['objective'] is None


def test_factorial_prompt_off_uses_same_model_control_not_cached_score():
    row = fixture_row()
    rules = 'XOR 2 1 2\nOROUT 2 3 1\nOROUT 2 6 2\nXOR 2 3 9'
    ann = {'r': {'original_rebuttal_probability': .01, 'rebuttal_probability': 1.}}
    assert xor(row, rules)['prediction'] == 'B'  # published cached baseline
    assert xor(row, rules, ann, corrected=False)['prediction'] == 'A'
    assert xor(row, rules, ann, corrected=True)['prediction'] == 'B'
