from types import SimpleNamespace
import pytest
from delta_cli.public_eval.bfcl_errors import ContextExhausted
from delta_cli.public_eval.tools import infer_entry


def test_context_exhaustion_is_native_case_failure():
    def infer(row,**kwargs):raise ContextExhausted(34820,34816)
    row=infer_entry(SimpleNamespace(inference=infer),{'id':'multi_turn_base_0'})
    assert row['result']=='Error during inference: BFCL context exhausted'
    assert row['error_type']=='context_exhausted'
    assert row['prompt_tokens']==34820 and row['context_limit']==34816


@pytest.mark.parametrize('error',[ValueError('BFCL context exhausted'),RuntimeError('server unavailable'),TimeoutError('timeout')])
def test_other_errors_still_fail_the_run(error):
    def infer(row,**kwargs):raise error
    with pytest.raises(type(error)):infer_entry(SimpleNamespace(inference=infer),{'id':'multi_turn_base_0'})


def test_success_retains_native_metadata():
    def infer(row,**kwargs):return [['call()']],{'input_token_count':[[10]]}
    assert infer_entry(SimpleNamespace(inference=infer),{'id':'case'})=={'id':'case','result':[['call()']],'input_token_count':[[10]]}


def test_native_scorer_rejects_exhaustion():
    pytest.importorskip('bfcl_eval.eval_checker.eval_runner',exc_type=ImportError)
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
    result=_evaluate_single_multi_turn_entry(None,'multi_turn_base_0','Error during inference: BFCL context exhausted',[['call()']],{'function':[]},'candidate','multi_turn_base')
    assert result['valid'] is False and result['error']['error_type']=='multi_turn:inference_error'


def test_handler_raises_typed_limit_before_request():
    pytest.importorskip('bfcl_eval.eval_checker.eval_runner',exc_type=ImportError)
    import threading
    from delta_cli.public_eval.bfcl_handler import Qwen35Handler
    handler=object.__new__(Qwen35Handler)
    handler._tool_schemas=threading.local()
    handler.validation_generation={'max_tokens':8192,'enable_thinking':True}
    handler.max_context_length=8
    handler.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:'prompt',tokenize=lambda p:list(range(10)))
    with pytest.raises(ContextExhausted) as error:handler._query_prompting({'message':[],'function':[]})
    assert error.value.prompt_tokens==10 and error.value.context_limit==8


def test_native_shard_keeps_all_exhausted_cases_in_denominator(tmp_path,monkeypatch):
    pytest.importorskip('bfcl_eval.eval_checker.eval_runner',exc_type=ImportError)
    from delta_cli.public_eval import tools
    from delta_cli.public_eval.suite import manifest
    from delta_cli.public_eval.coverage import expand,slice_suite
    suite=slice_suite(expand(manifest(),'bfcl_multi_turn'),'bfcl_multi_turn',[0,8])
    def infer(row,**kwargs):raise ContextExhausted(35000,34816)
    monkeypatch.setattr(tools,'handler_for',lambda *a,**kw:(SimpleNamespace(inference=infer),'DeltaQwen35Validation'))
    result=tools.run(suite,'http://unused','candidate','unused',tmp_path/'report.json',concurrency=4)
    assert sum(v['count'] for v in result['scores'].values())==8
    assert sum(v['context_exhausted'] for v in result['scores'].values())==8
    assert all(v['accuracy']==0 for v in result['scores'].values())
