import json
from types import SimpleNamespace as Obj

import pytest

from quote_power.cli import evaluate
from quote_power.data import write
from quote_power.scoring import Scorer


def test_api_cache_reuses_only_matching_requests(tmp_path, monkeypatch):
    import openai
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        tokens = [Obj(token='Yes', logprob=-.2), Obj(token=' No', logprob=-2.)]
        return Obj(choices=[Obj(message=Obj(content='Yes'), logprobs=Obj(content=[Obj(top_logprobs=tokens)]))])
    monkeypatch.setenv('LLM_API_KEY', 'test-placeholder')
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: Obj(base_url='https://test.invalid/v1', chat=Obj(completions=Obj(create=create))))
    scorer = Scorer('test', tmp_path)
    first = scorer.request('question', 'yesno')
    assert .5 < first['probability'] < 1
    assert scorer.request('question', 'yesno') == first
    assert len(calls) == 1
    scorer.request('different question', 'yesno')
    assert len(calls) == 2
    with pytest.raises((json.JSONDecodeError, ValueError)):
        scorer.request('bad JSON response', 'quote')
    assert len(list(tmp_path.glob('*.json'))) == 2


def test_factorial_full_pipeline(tmp_path):
    # Synthetic fixture, kept out of scientific data/results.
    row = dict(id='s', transcript='A: Decide now.\nB: Okay.', label='A', relations=[], arguments=[
        dict(id='s:a', variable=2, side='A', kind='reason', text='A decides', validity=0, grounding=0, target=None),
        dict(id='s:b', variable=5, side='B', kind='reason', text='B agrees', validity=0, grounding=-1, target=None)])
    write(tmp_path / 'data.jsonl', [row])
    write(tmp_path / 'annotations.jsonl', [dict(id=a['id'], quote='Decide now.', evidence_speaker='A', supports=True, attribution_correct=True) for a in row['arguments']])
    write(tmp_path / 'zero.jsonl', [dict(id='s', prediction='A')])
    (tmp_path / 'rules.txt').write_text('XOR 2 1 2\nOROUT 2 3 1\nOROUT 2 6 2\n')
    args = Obj(data=tmp_path / 'data.jsonl', rules=tmp_path / 'rules.txt', annotations=tmp_path / 'annotations.jsonl',
               factorial=True, zero_shot=tmp_path / 'zero.jsonl', output=tmp_path / 'out')
    evaluate(args)
    report = json.loads((tmp_path / 'out/report.json').read_text())
    assert len(report['methods']) == 12  # 8 factorial + 3 cached + zero-shot
    assert report['snippets'] == 1
    assert all(v['accuracy'] == 1 for v in report['methods'].values())


def test_purdue_null_backoff_and_endpoint(tmp_path, monkeypatch):
    import openai
    import quote_power.scoring as scoring
    responses = iter([None, Obj(choices=[])])
    options, pauses = {}, []
    def client(**kwargs):
        options.update(kwargs)
        return Obj(base_url=kwargs['base_url'], chat=Obj(completions=Obj(create=lambda **kw: next(responses))))
    monkeypatch.setenv('LLM_API_KEY', 'test-placeholder')
    monkeypatch.setenv('LLM_BASE_URL', 'https://genai.rcac.purdue.edu/api')
    monkeypatch.setattr(openai, 'OpenAI', client)
    monkeypatch.setattr(scoring.time, 'sleep', pauses.append)
    scorer = Scorer('test', tmp_path)
    assert options['base_url'] == 'https://genai.rcac.purdue.edu/api'
    assert options['max_retries'] == 0
    with pytest.raises(ValueError, match='no completion choices'):
        scorer.request('test', 'yesno')
    assert 5 in pauses  # documented null response triggers bounded backoff
    assert not list(tmp_path.glob('*.json'))
