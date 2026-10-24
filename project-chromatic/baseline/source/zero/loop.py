"""Finite, restartable selfplay -> train -> arena generations. No teacher dependency."""
from pathlib import Path
import json
import shutil
from .arena import arena
from .checkpoint import digest
from .data import atomic_json
from .model import load_model
from .selfplay import selfplay
from .train import train


def loop(executable, initial, output, iterations=1, games=8, simulations=128,
         steps=200, batch_size=64, pairs=8, max_plies=512, seed=1,
         replay_generations=4, threshold=0.55, start_fen=None, opening_fens=None):
    if iterations < 1 or replay_generations < 1:
        raise ValueError('positive iterations and replay_generations required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    settings = {'initial': digest(initial), 'executable': digest(executable), 'games': games,
                'simulations': simulations, 'steps': steps, 'batch_size': batch_size,
                'pairs': pairs, 'max_plies': max_plies, 'seed': seed,
                'replay_generations': replay_generations, 'threshold': threshold,
                'start_fen': start_fen, 'opening_fens': opening_fens}
    path = root / 'state.json'
    if path.exists():
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['settings'] != settings:
            raise ValueError('loop settings changed; use a new run directory')
    else:
        # This immutable initial snapshot makes restarts independent of caller modifications.
        model, _, actual = load_model(initial)
        initial_copy = root / 'initial.pt'
        shutil.copyfile(actual, initial_copy)
        model.export(root / 'initial.zero')
        state = {'settings': settings, 'completed': 0, 'champion': 'initial.pt', 'reports': []}
        atomic_json(path, state)
    for iteration in range(state['completed'], iterations):
        stage = root / f'generation_{iteration:04d}'
        stage.mkdir(exist_ok=True)
        champion = root / state['champion']
        model, _, _ = load_model(champion)
        champion_zero = stage / 'source.zero'
        model.export(champion_zero)
        kwargs = {'start_fen': start_fen} if start_fen else {}
        selfplay(executable, champion_zero, stage / 'selfplay', games=games,
                 simulations=simulations, seed=seed + iteration * 10000019,
                 max_plies=max_plies, **kwargs)
        replay = []
        for generation in range(max(0, iteration - replay_generations + 1), iteration + 1):
            replay.extend(sorted((root / f'generation_{generation:04d}' / 'selfplay').glob('game_*.jsonl')))
        if not any(p.stat().st_size for p in replay):
            raise RuntimeError('no completed selfplay games; increase --max-plies or use a stronger initial model')
        candidate = stage / 'candidate.pt'
        resume = candidate if candidate.exists() or Path(str(candidate) + '.prev').exists() else None
        train(replay, candidate, steps=steps, batch_size=batch_size, seed=seed + iteration,
              init=None if resume else champion, resume=resume, save_every=max(1, steps // 4))
        gate = stage / 'arena.json'
        if gate.exists():
            report = json.loads(gate.read_text(encoding='utf-8'))
            if report['candidate'] != digest(candidate.with_suffix('.zero')) or report['incumbent'] != digest(champion_zero):
                raise ValueError('arena report model mismatch')
        else:
            report = arena(executable, candidate.with_suffix('.zero'), champion_zero, gate,
                           pairs=pairs, simulations=simulations, max_plies=max_plies,
                           seed=seed + iteration, threshold=threshold, opening_fens=opening_fens)
        if report['promote']:
            state['champion'] = candidate.relative_to(root).as_posix()
        state['reports'].append({'iteration': iteration, 'arena': gate.relative_to(root).as_posix(),
                                 'promoted': report['promote']})
        state['completed'] = iteration + 1
        atomic_json(path, state)
    return state
