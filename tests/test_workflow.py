import json
from types import SimpleNamespace as Obj

import pytest

from quote_power.cli import evaluate
from quote_power.data import write
from quote_power.scoring import Scorer


def test_api_cache_reuses_only_matching_requests(tmp_path, monkeypatch):
    import openai
    import quote_power.scoring as scoring
    monkeypatch.setattr(scoring.time, 'sleep', lambda _: None)
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
    assert calls[2]['response_format']['json_schema']['strict'] is True
    assert calls[2]['response_format']['json_schema']['schema']['additionalProperties'] is False
    assert calls[-1]['response_format']['type'] == 'json_object'
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
    assert report['quote_checks']['all']['total'] == 2
    assert report['quote_checks']['all']['normalized_span_matches'] == 2
    assert report['quote_checks']['all']['retained'] == 2
    assert report['quote_checks']['rebuttal']['retention_fraction'] is None
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


def test_truncated_quote_retries_without_dropping_argument(tmp_path, monkeypatch):
    import openai
    import quote_power.scoring as scoring
    calls = []
    valid = dict(quote='NONE', evidence_speaker='A', supports=False, attribution_correct=True)
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return Obj(choices=[Obj(message=Obj(content='{"quote":'), finish_reason='length')])
        return Obj(choices=[Obj(message=Obj(content=json.dumps(valid)), finish_reason='stop')])
    monkeypatch.setenv('LLM_API_KEY', 'test-placeholder')
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: Obj(base_url='https://test.invalid/v1', chat=Obj(completions=Obj(create=create))))
    monkeypatch.setattr(scoring.time, 'sleep', lambda _: None)
    result = Scorer('test', tmp_path).request('quote prompt', 'quote')
    assert result['quote'] == 'NONE'
    assert calls[1]['max_tokens'] == 2 * calls[0]['max_tokens']
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_original_prompt_control_excludes_rebuttal_text():
    from quote_power.scoring import original_rebuttal_prompt
    row = dict(transcript='A: I decide.\nB: Okay.', arguments=[dict(id='a', text='A directs.')])
    arg = dict(target='a', text='A was only suggesting.')
    prompt = original_rebuttal_prompt(row, arg)
    assert 'A directs.' in prompt
    assert 'A was only suggesting.' not in prompt
    assert 'When uncertain, output No.' in prompt


def test_openwebui_connection_error_is_retried_but_invalid_requests_are_not(tmp_path, monkeypatch):
    import httpx
    import openai
    import quote_power.scoring as scoring
    calls, sleeps = [], []
    request = httpx.Request('POST', 'https://test.invalid/chat/completions')
    response = httpx.Response(400, request=request)
    body = {'detail': 'Open WebUI: Server Connection Error'}
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise openai.BadRequestError('backend unavailable', response=response, body=body)
        return Obj(choices=[])
    monkeypatch.setenv('LLM_API_KEY', 'test-placeholder')
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: Obj(base_url='https://test.invalid/v1', chat=Obj(completions=Obj(create=create))))
    monkeypatch.setattr(scoring.time, 'sleep', sleeps.append)
    scorer = Scorer('test', tmp_path)
    with pytest.raises(ValueError, match='no completion choices'):
        scorer.request('test', 'yesno')
    assert len(calls) == 2 and 5 in sleeps
    calls.clear()
    body['detail'] = 'Invalid request parameter'
    with pytest.raises(openai.BadRequestError):
        scorer.request('invalid', 'yesno')
    assert len(calls) == 1
    assert not list(tmp_path.glob('*.json'))


def test_score_completes_all_zero_shot_calls_before_argument_pool(tmp_path, monkeypatch):
    import quote_power.scoring as scoring
    from quote_power.data import load
    events = []
    class FakeScorer:
        def __init__(self, *args, **kwargs):
            self.client = Obj(base_url='https://test.invalid/')
        def request(self, prompt, kind):
            events.append((kind, prompt))
            if kind == 'quote':
                return dict(quote='NONE', evidence_speaker='A', supports=False, attribution_correct=True)
            return dict(probability=.6)
    monkeypatch.setattr(scoring, 'Scorer', FakeScorer)
    rows = [dict(id=str(i), transcript='A: Decide.\nB: Okay.', arguments=[
        dict(id=f'{i}:a', kind='reason', text='A decides.', side='A', target=None),
        dict(id=f'{i}:r', kind='rebuttal', text='A only suggests.', side='A', target=f'{i}:a')]) for i in range(2)]
    scoring.score(rows, 'test', tmp_path/'cache', tmp_path/'out')
    assert len(events) == 10
    assert all(kind == 'yesno' and 'Does speaker A' in prompt for kind, prompt in events[:2])
    assert events[2][0] == 'quote'
    assert len(load(tmp_path/'out/zero_shot.jsonl')) == 2
    annotations = load(tmp_path/'out/annotations.jsonl')
    assert len(annotations) == 4
    assert all(a['original_rebuttal_probability'] == .6 for a in annotations if a['id'].endswith(':r'))


def test_compact_quote_recovery_falls_back_without_inventing_judgments(tmp_path, monkeypatch):
    import openai
    import quote_power.scoring as scoring
    calls = []
    valid = dict(quote='A: Decide.', evidence_speaker='A', supports=True, attribution_correct=True)
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            return Obj(choices=[Obj(message=Obj(content='{"quote":"A: Decide."\n\n'), finish_reason='length')])
        return Obj(choices=[Obj(message=Obj(content=json.dumps(valid)), finish_reason='stop')])
    monkeypatch.setenv('LLM_API_KEY', 'test-placeholder')
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: Obj(base_url='https://test.invalid/', chat=Obj(completions=Obj(create=create))))
    monkeypatch.setattr(scoring.time, 'sleep', lambda _: None)
    result = Scorer('test', tmp_path).request('same evidence question', 'quote')
    assert len(calls) == 3
    assert calls[0]['response_format']['type'] == 'json_schema'
    assert 'compact JSON' in calls[1]['messages'][0]['content']
    assert calls[2]['response_format']['type'] == 'json_object'
    assert all(c['messages'][1]['content'] == 'same evidence question' for c in calls)
    assert result['quote'] == valid['quote'] and result['supports'] is True
    assert result['_request_metadata']['quote_formatting_attempts'] == 3
    assert len(list(tmp_path.glob('*.json'))) == 1
    assert len(list((tmp_path/'errors').glob('*.json'))) == 2


def test_cache_isolated_by_model_and_endpoint(tmp_path, monkeypatch):
    import openai
    import quote_power.scoring as scoring
    calls = []
    def client(**options):
        def create(**payload):
            calls.append((options['base_url'], payload['model']))
            tokens = [Obj(token='Yes', logprob=-.2), Obj(token='No', logprob=-2.)]
            return Obj(choices=[Obj(message=Obj(content='Yes'), logprobs=Obj(content=[Obj(top_logprobs=tokens)]))])
        return Obj(base_url=options['base_url'], chat=Obj(completions=Obj(create=create)))
    monkeypatch.setenv('LLM_API_KEY', 'test-placeholder')
    monkeypatch.setattr(openai, 'OpenAI', client)
    monkeypatch.setattr(scoring.time, 'sleep', lambda _: None)
    for model, endpoint in [('m1', 'https://one.invalid/'), ('m1', 'https://one.invalid/'),
                            ('m2', 'https://one.invalid/'), ('m1', 'https://two.invalid/')]:
        Scorer(model, tmp_path, endpoint).request('same question', 'yesno')
    assert len(calls) == 3
    assert len(list(tmp_path.glob('*.json'))) == 3


def test_quote_only_reuses_successes_and_preserves_partial_factorial(tmp_path, monkeypatch):
    import openai
    import quote_power.scoring as scoring
    from quote_power.data import load
    calls = []
    def create(**payload):
        calls.append(payload)
        if 'response_format' in payload:
            content = json.dumps(dict(quote='NONE', evidence_speaker='A', supports=False, attribution_correct=True))
            return Obj(choices=[Obj(message=Obj(content=content))])
        tokens = [Obj(token='Yes', logprob=-.2), Obj(token='No', logprob=-2.)]
        return Obj(choices=[Obj(message=Obj(content='Yes'), logprobs=Obj(content=[Obj(top_logprobs=tokens)]))])
    monkeypatch.setenv('LLM_API_KEY', 'test-placeholder')
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: Obj(base_url='https://one.invalid/', chat=Obj(completions=Obj(create=create))))
    monkeypatch.setattr(scoring.time, 'sleep', lambda _: None)
    row = dict(id='s', transcript='A: Decide.\nB: Okay.', arguments=[
        dict(id='a', kind='reason', text='A decides.', side='A', target=None),
        dict(id='r', kind='rebuttal', text='A only suggests.', side='A', target='a')])
    output, cache = tmp_path/'out', tmp_path/'cache'
    scoring.score([row], 'm', cache, output)
    original_export = (output/'annotations.jsonl').read_bytes()
    assert len(calls) == 5
    # Two cached quotes and zero-shot are reused; only a new argument needs a call.
    row['arguments'].append(dict(id='b', kind='reason', text='B agrees.', side='B', target=None))
    scoring.score([row], 'm', cache, output, quote_only=True)
    assert len(calls) == 6 and 'response_format' in calls[-1]
    assert (output/'annotations.jsonl').read_bytes() == original_export
    quotes = load(output/'quote_annotations.jsonl')
    assert len(quotes) == 3
    assert all('rebuttal_probability' not in a and 'original_rebuttal_probability' not in a for a in quotes)


def test_quote_only_evaluation_holds_cached_scores_fixed(tmp_path):
    row = dict(id='s', transcript='A: Decide now.\nB: Okay.', label='A', relations=[], arguments=[
        dict(id='a', variable=2, side='A', kind='reason', text='A decides', validity=0, grounding=0, target=None),
        dict(id='b', variable=5, side='B', kind='reason', text='B agrees', validity=0, grounding=-1, target=None),
        dict(id='r', variable=8, side='A', kind='rebuttal', text='Only suggests', validity=0, grounding=-1, target='a')])
    write(tmp_path/'data.jsonl', [row])
    # Poisoned score fields must be ignored in the fixed-score comparison.
    annotations = [dict(id=a['id'], quote='Decide now.', evidence_speaker='A', supports=True,
                        attribution_correct=True, original_rebuttal_probability='invalid',
                        rebuttal_probability='invalid') for a in row['arguments']]
    write(tmp_path/'ann.jsonl', annotations)
    (tmp_path/'rules.txt').write_text('XOR 2 1 2\nOROUT 2 3 1\nOROUT 2 6 2\n')
    args = Obj(data=tmp_path/'data.jsonl', rules=tmp_path/'rules.txt', annotations=tmp_path/'ann.jsonl',
               factorial=False, quote_only=True, zero_shot=None, output=tmp_path/'out')
    evaluate(args)
    report = json.loads((tmp_path/'out/report.json').read_text())
    assert report['scope'] == 'quote_filter_only'
    assert len(report['methods']) == 5
    assert 'grounded_cached' in report['methods']['grounded_quote1_cached_scores']['comparisons']
    assert report['methods']['xor_quote1_cached_scores']['accuracy'] == report['methods']['xor_cached']['accuracy']
    write(tmp_path/'ann.jsonl', annotations[:-1])
    with pytest.raises(ValueError, match='cover exactly'):
        evaluate(args)
