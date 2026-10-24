from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from . import checkpoint
from .data import atomic_json, tensors
from .model import ModelConfig, PolicyWDL


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def initialize(path, seed=1, width=64, blocks=2):
    seed_all(seed)
    model = PolicyWDL(ModelConfig(width, blocks))
    generator = torch.Generator().manual_seed(seed)
    checkpoint.save(path, {'config': asdict(model.config), 'model': model.state_dict(),
                          'optimizer': None, 'step': 0, 'generator': generator.get_state(),
                          'dataset': [], 'settings': {}, 'seed': seed})
    model.export(Path(path).with_suffix('.zero'))
    return model


def legacy_train(paths, output, steps=1000, batch_size=64, lr=0.001, seed=1,
          resume=None, init=None, width=64, blocks=2, save_every=100, device='cpu'):
    if steps < 0 or batch_size < 1 or save_every < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('invalid training settings')
    if resume and init:
        raise ValueError('choose resume or init')
    torch.set_num_threads(1)
    seed_all(seed)
    dataset = [{'sha256': checkpoint.digest(p), 'bytes': Path(p).stat().st_size} for p in paths]
    x, p, v, mask = tensors(paths)
    settings = {'batch_size': batch_size, 'lr': lr, 'device': device, 'decay': 0.999,
                'torch': str(torch.__version__)}
    state, recovered = (checkpoint.load(resume or init) if resume or init else (None, None))
    config = ModelConfig(**state['config']) if state else ModelConfig(width, blocks)
    model = PolicyWDL(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    start = 0
    if state:
        model.load_state_dict(state['model'])
        if resume:
            if dataset != state['dataset'] or settings != state['settings']:
                raise ValueError('resume requires identical dataset and training settings; use --init for a new generation')
            if state['optimizer'] is None:
                raise ValueError('no optimizer in initial checkpoint; use --init')
            optimizer.load_state_dict(state['optimizer'])
            generator.set_state(state['generator'])
            start = state['step']
            checkpoint.restore_rng(state['rng'])
    if steps < start:
        raise ValueError('steps is a total target, smaller than checkpoint step')
    history = []
    def save(step):
        checkpoint.save(output, {'config': asdict(config), 'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'step': step,
                                'generator': generator.get_state(), 'dataset': dataset,
                                'settings': settings, 'seed': seed})
        model.export(Path(output).with_suffix('.zero'))
    model.train()
    for step in range(start, steps):
        ids = torch.randint(len(x), (batch_size,), generator=generator)
        xb, pb, vb, mb = (item[ids].to(device) for item in (x, p, v, mask))
        for group in optimizer.param_groups:
            group['lr'] = lr * 0.999 ** step
        optimizer.zero_grad(set_to_none=True)
        policy, wdl = model(xb)
        policy_loss = -(pb * torch.log_softmax(policy.masked_fill(~mb, -1e9), dim=-1)).sum(-1).mean()
        value_loss = -(vb * torch.log_softmax(wdl, dim=-1)).sum(-1).mean()
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({'step': step + 1, 'policy_loss': float(policy_loss.detach()),
                        'wdl_loss': float(value_loss.detach())})
        if (step + 1) % save_every == 0:
            save(step + 1)
    if steps == start or steps % save_every:
        save(steps)
    report = {'records': len(x), 'start_step': start, 'step': steps,
              'recovered_from': str(recovered) if recovered else None,
              'device': device, 'last': history[-1] if history else None}
    atomic_json(Path(output).with_suffix('.train.json'), report)
    return report


# v0.1 exact continuation remains explicit; new training uses sparse bounded replay.
from .training import train
