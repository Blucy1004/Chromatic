"""Device training primitives; no source mutation and no hidden sampler RNG."""
from contextlib import nullcontext
import time
import torch
from .devices import assert_finite,NumericalError,BatchTransfer,cpu_state,clip_gradients
from .model_adapter import losses_from_batch

class Execution:
    def __init__(self,model,optimizer,scheduler,runtime,profile,config,start_step=0):
        self.model=model;self.optimizer=optimizer;self.scheduler=scheduler
        self.runtime=runtime;self.profile=profile;self.config=config
        self.transfer=BatchTransfer(runtime,preallocate=config.get('transfer_mode','copy')=='preallocated')
        self.samples=0;self.updates=0;self.compute_seconds=0.;self.transfer_seconds=0.
        self.fallbacks=[]
        self.precision_checks=[]
        self.start_step=int(start_step)

    def span(self,stage,**kwargs):
        return self.profile.span(stage,sync=True,**kwargs) if self.profile else nullcontext()

    def to_device(self,tensors):
        # Synchronization here is intentional: transfer timings must not measure
        # merely queue submission. Resident-cache batches bypass large copies.
        if self.runtime.device.type=='cpu':return tensors
        self.runtime.synchronize();start=time.perf_counter()
        with self.span('host_to_device'):
            output=self.transfer(tensors)
        self.runtime.synchronize();self.transfer_seconds+=time.perf_counter()-start
        return output

    def step(self,tensors):
        self.runtime.synchronize();started=time.perf_counter()
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        n=len(tensors[0]);accum=min(n,self.config.get('gradient_accumulation',1))
        ranges=[(n*i//accum,n*(i+1)//accum) for i in range(accum)]
        reference_loss=None
        if self.runtime.precision=='bf16' and (self.start_step+self.updates)%32==0:
            with torch.no_grad(),self.span('precision_reference'):
                reference_loss=float(losses_from_batch(self.model,tensors)[0])
        summaries=[];weighted_loss=0.
        for lo,hi in ranges:
            micro=tuple(t[lo:hi] for t in tensors)
            with self.span('forward'):
                with self.runtime.autocast():loss,pl,vl,pw,vw,_=losses_from_batch(self.model,micro)
            if not bool(torch.isfinite(loss)):
                if self.runtime.precision=='bf16':
                    # No optimizer step has happened: discard all partial grads
                    # and repeat the entire effective batch safely in FP32.
                    self.runtime.fallback_fp32('nonfinite BF16 loss before optimizer step')
                    self.fallbacks.append({'reason':'nonfinite_loss','update':self.updates})
                    self.optimizer.zero_grad(set_to_none=True);return self.step(tensors)
                raise NumericalError('NaN/Inf loss; uncommitted shard retained')
            with self.span('backward'):(loss*((hi-lo)/n)).backward()
            weighted_loss=weighted_loss+loss.detach()*((hi-lo)/n)
            summaries.append(torch.stack([(pl.detach()*pw).sum(),pw.sum(),(vl.detach()*vw).sum(),vw.sum()]))
        if reference_loss is not None:
            observed=float(weighted_loss);delta=observed-reference_loss
            self.precision_checks.append({'update':self.updates,'fp32_loss':reference_loss,'bf16_loss':observed,'delta':delta})
            self.precision_checks=self.precision_checks[-64:]
            if delta>max(.02,abs(reference_loss)*.02):
                self.runtime.fallback_fp32('BF16 batch loss exceeded FP32 by the predefined 2% / 0.02 tolerance')
                self.fallbacks.append({'reason':'loss_deterioration','update':self.updates})
                self.optimizer.zero_grad(set_to_none=True)
                self.compute_seconds+=time.perf_counter()-started
                return self.step(tensors)
        try:
            with self.span('gradient_clip'):
                grad=clip_gradients(self.model.parameters(),1.)
        except RuntimeError:
            if self.runtime.precision!='bf16':raise
            self.runtime.fallback_fp32('nonfinite BF16 gradient before optimizer step')
            self.fallbacks.append({'reason':'nonfinite_gradient','update':self.updates})
            self.optimizer.zero_grad(set_to_none=True);return self.step(tensors)
        with self.span('optimizer'):
            self.optimizer.step();self.scheduler.step()
        with self.span('finite_check'):assert_finite(self.model,self.optimizer)
        self.runtime.synchronize();self.compute_seconds+=time.perf_counter()-started
        self.samples+=n;self.updates+=1
        # Keep metrics on-device until a checkpoint/epoch boundary; no six
        # scalar host synchronizations per update.
        self.last_metrics=(torch.stack(summaries).sum(0),grad.detach())
        return self.last_metrics

    def metrics(self):
        if not hasattr(self,'last_metrics'):return {}
        values,grad=self.last_metrics;pn,pw,vn,vw=values.detach().cpu().tolist()
        return {'policy_loss':pn/pw if pw else None,'wdl_loss':vn/vw if vw else None,
                'gradient_norm':float(grad.cpu()),'learning_rate':self.optimizer.param_groups[0]['lr']}

    def snapshot(self):
        return {'training_samples':self.samples,'training_updates_this_process':self.updates,
                'pure_training_seconds':self.compute_seconds,'host_to_device_seconds':self.transfer_seconds,
                'training_samples_per_second':self.samples/self.compute_seconds if self.compute_seconds else None,
                'precision_fallbacks':self.fallbacks,'precision_checks':self.precision_checks,'effective_precision':self.runtime.precision}
