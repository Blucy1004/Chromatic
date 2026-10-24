import argparse
import json
from pathlib import Path
from .train import initialize, train, legacy_train
from .model import load_model
from .selfplay import selfplay
from .bootstrap import bootstrap
from .arena import arena
from .loop import loop
from .verify import verify
from .champion import rollback


def main():
    parser = argparse.ArgumentParser(description='PROJECT ZERO training and evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('init')
    p.add_argument('--output', required=True); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--width', type=int, default=64); p.add_argument('--blocks', type=int, default=2)
    p = commands.add_parser('export')
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p = commands.add_parser('verify')
    p.add_argument('--executable', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--positions', type=int, default=100)
    p.add_argument('--seed', type=int, default=2026); p.add_argument('--tolerance', type=float, default=2e-5)
    p.add_argument('--backend',choices=['Scalar','SIMD'],default='Scalar')
    p = commands.add_parser('bootstrap')
    p.add_argument('--teacher', required=True); p.add_argument('--output', required=True)
    p.add_argument('--positions', type=int, default=128); p.add_argument('--nodes', type=int, default=2000)
    p.add_argument('--multipv', type=int, default=4); p.add_argument('--seed', type=int, default=1)
    p = commands.add_parser('train')
    p.add_argument('--data', nargs='+', required=True); p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=1000); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001); p.add_argument('--seed', type=int, default=1)
    source = p.add_mutually_exclusive_group(); source.add_argument('--resume'); source.add_argument('--init')
    p.add_argument('--width', type=int, default=64); p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--save-every', type=int, default=100); p.add_argument('--device', default='cpu')
    p.add_argument('--capacity',type=int,default=10000);p.add_argument('--recent-fraction',type=float,default=.75)
    p.add_argument('--validation-fraction',type=float,default=.15);p.add_argument('--optimizer-name',choices=['adamw','sgd'],default='adamw')
    p.add_argument('--legacy',action='store_true',help='explicit v0.1 optimizer continuation')
    p = commands.add_parser('selfplay')
    p.add_argument('--executable', required=True); p.add_argument('--model'); p.add_argument('--output', required=True)
    p.add_argument('--games', type=int, default=8); p.add_argument('--simulations', type=int, default=128)
    p.add_argument('--seed', type=int, default=1); p.add_argument('--max-plies', type=int, default=512)
    import chess
    p.add_argument('--start-fen', default=chess.STARTING_FEN); p.add_argument('--noise', type=float, default=0.25)
    p.add_argument('--temperature-plies', type=int, default=30)
    p.add_argument('--temperature',type=float,default=1);p.add_argument('--late-temperature',type=float,default=0)
    p.add_argument('--opening-plies',type=int,default=0);p.add_argument('--resume-truncated',action='store_true')
    p = commands.add_parser('arena')
    p.add_argument('--executable', required=True); p.add_argument('--candidate', required=True)
    p.add_argument('--incumbent', required=True); p.add_argument('--output', required=True)
    p.add_argument('--pairs', type=int, default=8); p.add_argument('--simulations', type=int, default=128)
    p.add_argument('--max-plies', type=int, default=512); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--threshold', type=float, default=0.55); p.add_argument('--confidence', type=float, default=0.95)
    p.add_argument('--candidate-executable');p.add_argument('--incumbent-executable');p.add_argument('--milliseconds',type=int)
    p.add_argument('--opening-suite');p.add_argument('--candidate-options',type=json.loads);p.add_argument('--incumbent-options',type=json.loads)
    p.add_argument('--sprt',action='store_true')
    p = commands.add_parser('loop')
    p.add_argument('--executable', required=True); p.add_argument('--initial', required=True)
    p.add_argument('--output', required=True); p.add_argument('--iterations', type=int, default=1)
    p.add_argument('--games', type=int, default=8); p.add_argument('--simulations', type=int, default=128)
    p.add_argument('--steps', type=int, default=200); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--pairs', type=int, default=8); p.add_argument('--max-plies', type=int, default=512)
    p.add_argument('--seed', type=int, default=1); p.add_argument('--replay-generations', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.55); p.add_argument('--start-fen')
    p.add_argument('--engine-options',type=json.loads);p.add_argument('--opening-plies',type=int,default=6)
    p.add_argument('--arena-milliseconds',type=int);p.add_argument('--opening-suite')
    p=commands.add_parser('rollback');p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'init':
        initialize(args.pop('output'), **args); result = {'initialized': True}
    elif command == 'export':
        model, _, actual = load_model(args['checkpoint']); model.export(args['output'])
        result = {'checkpoint': str(actual), 'export': args['output']}
    elif command == 'train':
        # glob also accepts absolute paths on Windows.
        import glob
        paths = [path for pattern in args.pop('data') for path in sorted(glob.glob(pattern))]
        if not paths:
            raise ValueError('no data files matched')
        legacy=args.pop('legacy')
        if legacy:
            for name in ('capacity','recent_fraction','validation_fraction','optimizer_name'):args.pop(name)
        result = (legacy_train if legacy else train)(paths, **args)
    else:
        result = {'selfplay': selfplay, 'bootstrap': bootstrap, 'arena': arena, 'loop': loop, 'verify': verify,'rollback':rollback}[command](**args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
