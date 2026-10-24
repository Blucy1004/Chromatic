import copy
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shardtrain.devices import (DeviceRuntime,DeviceUnavailable,NumericalError,cpu_state,assert_finite,
    optimizer_to_device,make_optimizer,hardware_signature,BatchTransfer)

def test_cpu_runtime_state_and_independent_portable_copy():
    r=DeviceRuntime('cpu','fp32',1);r.seed(31);saved=r.capture_rng()
    expected=torch.rand(7);r.restore_rng(saved);assert torch.equal(expected,torch.rand(7))
    a=torch.tensor([1.],requires_grad=True);b=cpu_state({'x':[a]})
    a.data.fill_(9);assert b['x'][0].item()==1 and not b['x'][0].requires_grad
    assert r.device.type=='cpu' and r.memory()['allocated_bytes']==0

def test_explicit_validation_and_auto_conservative_selection():
    with pytest.raises(ValueError):DeviceRuntime('bad')
    with pytest.raises(ValueError):DeviceRuntime('cpu','bf16')
    with pytest.raises(ValueError):DeviceRuntime('cpu',cpu_threads=0)
    r=DeviceRuntime('auto');assert r.device.type=='cpu' and 'conservative' in r.decision['reason']
    assert DeviceRuntime('cpu','bf16',fallback='cpu').precision=='fp32'

def test_optimizer_portability_and_nonfinite_detection():
    model=torch.nn.Linear(3,2);optimizer=make_optimizer(model,mode='loop')
    model(torch.ones(4,3)).sum().backward();optimizer.step();optimizer_to_device(optimizer,'cpu')
    assert_finite(model,optimizer,True)
    for state in optimizer.state.values():assert state['step'].device.type=='cpu'
    next(model.parameters()).data.flatten()[0]=float('nan')
    with pytest.raises(NumericalError):assert_finite(model,optimizer)

def test_signature_changes_for_architecture_and_cpu_buffers_are_identity():
    a=hardware_signature({'width':64,'blocks':2});b=hardware_signature({'width':128,'blocks':2})
    assert a['key']!=b['key'] and a['details']['policy_actions']==4672
    batch=(torch.ones(2,3),torch.zeros(2,3))
    assert BatchTransfer(DeviceRuntime('cpu'))(batch) is batch

def test_xpu_unavailable_error_or_explicit_fallback(monkeypatch):
    import shardtrain.devices as devices
    monkeypatch.setattr(devices,'available_xpu',lambda:(False,'unit-test unavailable'))
    with pytest.raises(DeviceUnavailable,match='unit-test'):DeviceRuntime('xpu')
    assert DeviceRuntime('xpu',fallback='cpu').device.type=='cpu'

def test_recommendation_rejects_stale_architecture_and_unmeasured_failures(tmp_path):
    from shardtrain.benchmarks import select_recommendation,read_recommendation
    from shardtrain.storage import atomic_json
    signature=hardware_signature({'width':64,'blocks':2})
    row={'status':'ok','all_losses_finite':True,'quality_relative_loss_delta':0.,'device':'cpu','precision':'fp32',
        'cpu_threads':1,'batch_size':64,'mode':'copy','optimizer':'default','accumulation':1,
        'seconds_median':1.,'samples_per_second_median':100.,'training_samples_per_second_median':101.,'repeats':3,'steps':15}
    failed={**row,'status':'error','samples_per_second_median':999.}
    drift={**row,'quality_relative_loss_delta':.5,'samples_per_second_median':998.}
    result=select_recommendation([row,failed,drift],signature)
    target=tmp_path/'recommendation.json';atomic_json(target,result)
    assert result['measurement']['samples_per_second_median']==100.
    assert read_recommendation(target,{'width':64,'blocks':2}) is not None
    assert read_recommendation(target,{'width':128,'blocks':2}) is None
    assert read_recommendation(target,{'width':64,'blocks':2},cpu_threads=2) is None
    assert read_recommendation(target,{'width':64,'blocks':2},precision='bf16') is None

def test_cpu_microbenchmark_finite_and_bounded():
    from shardtrain.benchmarks import benchmark_case
    from shardtrain.model_adapter import PolicyWDL,ModelConfig
    torch.manual_seed(4);model=PolicyWDL(ModelConfig(width=4,blocks=0))
    x=torch.rand(4,1280);p=torch.zeros(4,4672);p[:,0]=1
    v=torch.ones(4,3)/3;mask=torch.ones(4,4672,dtype=torch.bool);weights=torch.ones(4)
    result,state=benchmark_case({'config':{'width':4,'blocks':0},'model':cpu_state(model.state_dict())},
        (x,p,v,mask,weights,weights),batch_size=32,repeats=1,steps=2,warmup=1)
    assert result['all_losses_finite'] and result['samples_per_second_median']>0
    assert all(t.device.type=='cpu' for t in state.values())
    with pytest.raises(ValueError):benchmark_case({'config':{}},(),repeats=100)
