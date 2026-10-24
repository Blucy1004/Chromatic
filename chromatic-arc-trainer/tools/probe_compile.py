"""Isolated optional Windows/XPU torch.compile experiment with a hard timeout."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]

def worker(device):
    sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
    from shardtrain.model_adapter import torch,PolicyWDL,ModelConfig,checkpoint,losses_from_batch
    from shardtrain.devices import DeviceRuntime,cpu_state
    runtime=DeviceRuntime(device);runtime.seed(2026)
    initial,_=checkpoint.load(ROOT/'champion/model.pt')
    model=PolicyWDL(ModelConfig(**initial['config'])).to(runtime.device);model.load_state_dict(initial['model'])
    x=torch.rand(32,1280,device=runtime.device);p=torch.zeros(32,4672,device=runtime.device);p[:,0]=1
    v=torch.ones(32,3,device=runtime.device)/3;mask=torch.zeros(32,4672,dtype=torch.bool,device=runtime.device);mask[:,:20]=True
    w=torch.ones(32,device=runtime.device);tensors=x,p,v,mask,w,w
    loss=losses_from_batch(model,tensors)[0];loss.backward();reference=float(loss.detach());model.zero_grad(set_to_none=True)
    runtime.synchronize();started=time.perf_counter()
    compiled=torch.compile(model)
    actual=losses_from_batch(compiled,tensors)[0];actual.backward();runtime.synchronize()
    elapsed=time.perf_counter()-started;value=float(actual.detach())
    result={'status':'ok','device':device,'compile_forward_backward_seconds':elapsed,'eager_loss':reference,
        'compiled_loss':value,'absolute_loss_error':abs(value-reference),
        'gradients_finite':all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'default_enabled':False,
        'scope':'actual CHROMATIC 64x2 model and weighted-loss shape, one synthetic batch; separate optional experiment'}
    print(json.dumps(result));return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--device',choices=['cpu','xpu'],default='xpu');parser.add_argument('--timeout',type=float,default=15)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/benchmarks/compile-probe.json')
    args=parser.parse_args()
    if args.worker:
        try:return worker(args.device)
        except Exception:
            print(json.dumps({'status':'error','device':args.device,'error':traceback.format_exc(),'default_enabled':False}));return 1
    if not 1<=args.timeout<=60:raise ValueError('Compile probe timeout must be between 1 and 60 seconds')
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    scratch=ROOT.parents[1]/'work/arc-compile-probe-cache';scratch.mkdir(parents=True,exist_ok=True)
    env['TORCHINDUCTOR_CACHE_DIR']=str(scratch)
    started=time.perf_counter()
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--device',args.device],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    try:
        stdout,stderr=process.communicate(timeout=args.timeout)
        lines=[line for line in stdout.splitlines() if line.startswith('{')]
        result=json.loads(lines[-1]) if lines else {'status':'error','error':'worker did not emit a structured result'}
        result.update(returncode=process.returncode,stderr=stderr[-10000:])
    except subprocess.TimeoutExpired:
        # Stop the compiler's child tree too; no background compile is left.
        if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=5)
        else:process.kill()
        stdout,stderr=process.communicate(timeout=5)
        lines=[line for line in stdout.splitlines() if line.startswith('{')]
        result=json.loads(lines[-1]) if lines else {'status':'timeout',
            'meaning':'No success or unsupported claim; this environment did not finish the bounded optional compile experiment'}
        result.update(process_shutdown_timeout=True,timeout_seconds=args.timeout,stdout_tail=stdout[-4000:],stderr=stderr[-10000:])
    result.update(device=args.device,elapsed_seconds=time.perf_counter()-started,default_enabled=False,
        sources=['https://docs.pytorch.org/docs/main/notes/get_start_xpu.html','https://docs.pytorch.org/tutorials/unstable/inductor_windows.html'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
