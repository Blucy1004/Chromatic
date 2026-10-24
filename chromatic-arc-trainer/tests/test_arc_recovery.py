"""Small CPU unit checks; real XPU and child-process runs are explicit tools."""
from pathlib import Path
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('validate_arc',ROOT/'tools/validate_arc.py')
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)


def test_recursive_cpu_snapshot_is_detached_and_independent():
    a=torch.tensor([1.,2.],requires_grad=True);tree={'a':a,'b':(torch.ones(1),[np.arange(2)])}
    snapshot=validation.tensor_tree_cpu(tree);validation.assert_cpu_tensors(snapshot)
    assert not snapshot['a'].requires_grad
    with torch.no_grad():a[0]=99
    assert snapshot['a'][0].item()==1


def test_exact_semantic_comparison_and_explicit_float_tolerance():
    a={'model':torch.tensor([1.,2.]),'rng':torch.tensor([1,2],dtype=torch.uint8),'replay':[{'id':'a'}]}
    validation.tree_compare(a,validation.tensor_tree_cpu(a))
    b=validation.tensor_tree_cpu(a);b['model'][0]+=1e-6
    with pytest.raises(AssertionError,match='tensor differs'):validation.tree_compare(a,b)
    validation.tree_compare(a,b,atol=2e-6)
    b['rng'][0]=3
    with pytest.raises(AssertionError):validation.tree_compare(a,b,atol=2e-6)


def test_normalized_record_read_is_bounded_and_empty_rejected(tmp_path):
    path=tmp_path/'fixed.jsonl';path.write_text('\n'.join(json.dumps({'identity':str(i)}) for i in range(5)),encoding='utf-8')
    assert [r['identity'] for r in validation.read_records(path,2)]==['0','1']
    path.write_text('',encoding='utf-8')
    with pytest.raises(ValueError,match='fixed normalized'):validation.read_records(path)


def test_weighted_loss_equation_matches_preserved_adapter():
    from shardtrain.anchors import opening_anchors
    from shardtrain.model_adapter import PolicyWDL,ModelConfig,batch,weighted_losses
    torch.set_num_threads(1);torch.manual_seed(2026)
    rows=opening_anchors(ROOT/'fixtures',13,2026)[:2]
    rows=[dict(r,wdl=[.4,.2,.4],value_weight=.25) for r in rows]
    model=PolicyWDL(ModelConfig(width=8,blocks=1))
    old=weighted_losses(model,rows)[0]
    actual=validation._loss_outputs(model,batch(rows))[0]
    assert torch.equal(old,actual)
    step=validation._train_one({'config':{'width':8,'blocks':1},'model':model.state_dict()},rows,'cpu')
    reference=step['gradient_norm_reference_float64']
    assert abs(step['gradient_norm']-reference)<=max(1e-5,reference*1e-5)
    expected=min(reference,reference/(reference+1e-6))
    assert abs(step['clipped_gradient_norm_float64']-expected)<=1e-5
    assert step['head_raw_gradient_norm_float64']['policy']>0
    assert step['head_raw_gradient_norm_float64']['wdl']>0
    assert step['head_update_norm_float64']['policy']>0
    assert step['head_update_norm_float64']['wdl']>0


@pytest.mark.skipif(os.environ.get('CHROMATIC_RUN_XPU_TESTS')!='1',reason='explicit actual XPU regression')
def test_xpu_clip_norm_matches_independent_cpu_float64_reference():
    from shardtrain.devices import clip_gradients
    assert torch.xpu.is_available(),'Actual XPU explicitly requested but unavailable'
    generator=torch.Generator().manual_seed(2026)
    shapes=[(64,1280),(64,),(64,64),(64,),(64,64),(64,),(4672,64),(4672,),(3,64),(3,)]
    raw=[torch.randn(shape,generator=generator)*.01 for shape in shapes]
    reference=sum(float(g.double().square().sum()) for g in raw)**.5
    parameters=[]
    for gradient in raw:
        p=torch.nn.Parameter(torch.zeros_like(gradient,device='xpu'));p.grad=gradient.to('xpu');parameters.append(p)
    torch.xpu.synchronize()
    returned=clip_gradients(parameters,1.);torch.xpu.synchronize()
    actual=sum(float(p.grad.cpu().double().square().sum()) for p in parameters)**.5
    assert abs(float(returned.cpu())-reference)<=max(1e-5,reference*1e-5)
    assert abs(actual-min(reference,reference/(reference+1e-6)))<=1e-5

