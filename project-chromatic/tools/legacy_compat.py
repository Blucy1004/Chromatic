"""Reproduce one optimizer step with both readers, without mixing Python packages."""
from pathlib import Path
import json
import subprocess
import sys
import torch
root=Path(__file__).resolve().parents[1]
directory=root/'reports/compatibility';directory.mkdir(parents=True,exist_ok=True)
new=directory/'chromatic-legacy-resumed.pt'
subprocess.run([sys.executable,'-m','zero','train','--legacy','--data',str(root/'runs/bootstrap/teacher.jsonl'),
                '--output',str(new),'--resume',str(root/'runs/bootstrap/model.pt'),'--steps','201',
                '--batch-size','32','--seed','2026','--save-every','50'],cwd=root,check=True,capture_output=True,text=True)
sys.path.insert(0,str(root/'baseline/source'))
from zero.train import train
old=directory/'v0-source-resumed.pt'
train([root/'runs/bootstrap/teacher.jsonl'],old,steps=201,batch_size=32,seed=2026,save_every=50,
      resume=root/'runs/bootstrap/model.pt')
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
a=torch.load(old,weights_only=False,map_location='cpu');b=torch.load(new,weights_only=False,map_location='cpu')
result={'original_format':a['format'],'new_format':b['format'],'step':b['step'],
        'model_optimizer_sampler_equal':all(equal(a[k],b[k]) for k in ('model','optimizer','generator'))}
(directory/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not result['model_optimizer_sampler_equal']:raise SystemExit('legacy continuation differs')
