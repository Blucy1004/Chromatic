from pathlib import Path
import copy,sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shardtrain.devices import DeviceRuntime,cpu_state
from shardtrain.execution import Execution
from shardtrain.model_adapter import PolicyWDL,ModelConfig,batch
from shardtrain.anchors import opening_anchors

def backend(model,accumulation):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(.05,.999**step))
    return Execution(model,optimizer,scheduler,DeviceRuntime('cpu'),None,{'gradient_accumulation':accumulation})

def test_accumulation_preserves_effective_batch_update_with_float_tolerance():
    torch.manual_seed(2026)
    model=PolicyWDL(ModelConfig(8,1));other=copy.deepcopy(model)
    records=opening_anchors(ROOT/'fixtures',13,2026)[:7]
    tensors=batch(records);a=backend(model,1);b=backend(other,3)
    a.step(tensors);b.step(tensors)
    for name,left in model.state_dict().items():
        torch.testing.assert_close(left,other.state_dict()[name],atol=2e-6,rtol=2e-5)
    assert a.samples==b.samples==7 and a.updates==b.updates==1

def test_nonfinite_loss_cannot_advance_optimizer_or_mutate_model():
    torch.manual_seed(2026);model=PolicyWDL(ModelConfig(8,1));run=backend(model,1)
    tensors=list(batch(opening_anchors(ROOT/'fixtures',13,2026)[:2]))
    tensors[2].fill_(float('nan'));original=cpu_state(model.state_dict())
    with pytest.raises(RuntimeError,match='NaN/Inf'):run.step(tuple(tensors))
    assert run.updates==0 and not run.optimizer.state
    for name,value in model.state_dict().items():assert torch.equal(value,original[name])
