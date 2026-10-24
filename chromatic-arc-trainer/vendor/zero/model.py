from dataclasses import asdict, dataclass
from pathlib import Path
import os
import struct
import torch
from torch import nn
from .encoding import INPUTS, ACTIONS


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    blocks: int = 2


class PolicyWDL(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        if not 1 <= config.width <= 1024 or not 0 <= config.blocks <= 16:
            raise ValueError('unsupported architecture')
        self.config = config
        self.input = nn.Linear(INPUTS, config.width)
        self.residual = nn.ModuleList(nn.Linear(config.width, config.width) for _ in range(config.blocks))
        self.policy = nn.Linear(config.width, ACTIONS)
        self.wdl = nn.Linear(config.width, 3)

    def forward(self, x):
        x = torch.relu(self.input(x))
        for layer in self.residual:
            x = torch.relu(x + layer(x))
        return self.policy(x), self.wdl(x)

    def export(self, path):
        """Little endian FP32, row-major Linear weights, then bias, per layer."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        with temp.open('wb') as f:
            f.write(b'ZERO0001')
            f.write(struct.pack('<4I', INPUTS, self.config.width, self.config.blocks, ACTIONS))
            for layer in (self.input, *self.residual, self.policy, self.wdl):
                for tensor in (layer.weight, layer.bias):
                    array = tensor.detach().cpu().numpy().astype('<f4')
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError('nonfinite model weights')
                    f.write(array.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)


def load_model(path):
    # Training checkpoints are trusted local files; checkpoint.load validates format.
    from .checkpoint import load
    state, actual = load(path)
    model = PolicyWDL(ModelConfig(**state['config']))
    model.load_state_dict(state['model'])
    model.eval()
    return model, state, actual
