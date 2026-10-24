"""Fixed-budget in-process native timings; repeated raw samples retained."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zero.checkpoint import digest
from zero.data import atomic_json

def benchmark(executable, model, output, repeats=5, evaluations=1000, simulations=128, backend=None,head=None):
    if not 1 <= repeats <= 20:
        raise ValueError('repeats must be 1..20')
    samples=[]
    for _ in range(repeats):
        command=[str(Path(executable).resolve()),str(Path(model).resolve()),str(evaluations),str(simulations)]
        if backend:
            command.append(backend)
        if head:
            if not backend:raise ValueError('explicit backend required with head ablation')
            command.append(head)
        samples.append(json.loads(subprocess.check_output(command,text=True,timeout=60)))
    medians={key:statistics.median(row[key] for row in samples) for key in samples[0]
             if isinstance(samples[0][key],(int,float))}
    report={'executable_sha256':digest(executable),'model_sha256':digest(model),
            'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':__import__('os').cpu_count(),
            'config':{'repeats':repeats,'evaluations':evaluations,'simulations_per_position':simulations,'backend':backend,'head':head,'mate_proof':'disabled'},
            'medians':medians,'samples':samples}
    atomic_json(output,report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--model',required=True)
    p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--evaluations',type=int,default=1000);p.add_argument('--simulations',type=int,default=128)
    p.add_argument('--backend',choices=['scalar','simd']);p.add_argument('--head',choices=['full','legal']);args=vars(p.parse_args())
    print(json.dumps(benchmark(**args),indent=2))
