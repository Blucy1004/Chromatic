from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE='https://database.lichess.org/lichess_db_eval.jsonl.zst'

def parser():
    p=argparse.ArgumentParser(description='CC0 streaming, rotating CHROMATIC shard trainer')
    p.add_argument('--positions',type=int,default=100000);p.add_argument('--shard-size',type=int,default=10000)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--data-directory',type=Path,default=ROOT/'data');p.add_argument('--checkpoint-directory',type=Path,default=ROOT/'runs/default')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--keep-shards',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--max-disk-gb',type=float,default=4);p.add_argument('--max-memory-gb',type=float,default=4)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--source',default=DEFAULT_SOURCE)
    p.add_argument('--calibration',type=Path,default=ROOT/'fixtures/calibration.json')
    p.add_argument('--openings-directory',type=Path,default=ROOT/'fixtures')
    p.add_argument('--initial-checkpoint',type=Path,default=ROOT/'champion/model.pt')
    p.add_argument('--min-depth',type=int,default=12);p.add_argument('--min-knodes',type=int,default=10)
    p.add_argument('--max-download-mb',type=int,default=128)
    p.add_argument('--checkpoint-every',type=int,default=32);p.add_argument('--replay-size',type=int,default=512)
    p.add_argument('--similarity-cap',type=int,default=50)
    return p

def validate(args):
    from .storage import digest
    import math
    c=vars(args).copy()
    for key in ['positions','shard_size','epochs','batch_size','max_download_mb','checkpoint_every','replay_size','similarity_cap','min_depth','min_knodes']:
        if c[key]<1:raise ValueError(key+' must be positive')
    if c['batch_size']>4096 or c['shard_size']>100000:raise ValueError('batch_size <= 4096 and shard_size <= 100000 required for bounded buffers')
    for key in ['learning_rate','max_disk_gb','max_memory_gb']:
        if not math.isfinite(c[key]) or c[key]<=0:raise ValueError(key+' must be finite and positive')
    if c['learning_rate']>1:raise ValueError('learning_rate must be <= 1')
    if c['max_memory_gb']<.5:raise ValueError('MaxMemoryGB below 0.5 is insufficient for the CPU PyTorch runtime')
    if c['max_disk_gb']<.1:raise ValueError('MaxDiskGB below 0.1 cannot safely reserve mirrored checkpoints and rollback shards')
    for key in ['data_directory','checkpoint_directory','calibration','openings_directory','initial_checkpoint']:c[key]=str(Path(c[key]).resolve())
    for key in ['data_directory','checkpoint_directory']:
        path=Path(c[key])
        if path==ROOT/'champion' or path.is_relative_to(ROOT/'champion'):raise ValueError('Champion directory is immutable; choose a different output directory')
        for protected in ['project-chromatic','project-chromatic-depth','project-chromatic-memory','project-zero']:
            if path.is_relative_to(ROOT.parent/protected):raise ValueError('Existing engine directories are read-only; choose a separate training output directory')
    if not Path(c['initial_checkpoint']).is_file():raise ValueError('Initial checkpoint does not exist')
    c['initial_sha256']=digest(c['initial_checkpoint'])
    c['validation_size']=max(16,min(1000,c['positions']//10));c['test_size']=c['validation_size']
    c['trainer_version']=1;c['scheduler_spec']={'decay':.999,'minimum_multiplier':.05}
    c['max_decompressed_mb']=max(128,c['max_download_mb']*16)
    if c['source'].startswith('https://'):
        if c['source']!=DEFAULT_SOURCE:raise ValueError('Only the official Lichess evaluation URL is enabled')
        c['source_provenance']={'license':'CC0-1.0','url':DEFAULT_SOURCE,'version':'HTTP ETag/Last-Modified pinned at first read'}
    else:
        source=Path(c['source']).resolve();sidecar=Path(str(source)+'.source.json')
        if not source.is_file() or not sidecar.is_file():raise ValueError('Local source requires an official CC0 .source.json provenance sidecar')
        meta=json.loads(sidecar.read_text(encoding='utf-8'))
        if meta.get('license')!='CC0-1.0' or meta.get('url')!=DEFAULT_SOURCE:raise ValueError('Only official CC0 evaluation data is accepted')
        if meta.get('local_sha256')!=digest(source):raise ValueError('Local source hash differs from provenance')
        c['source']=str(source);c['source_provenance']=meta
    if not Path(c['calibration']).is_file():raise ValueError('Independent calibration artifact required; run tools/fit_calibration.py first')
    c['calibration_sha256']=digest(c['calibration'])
    if not (Path(c['openings_directory'])/'official-openings.source.json').is_file():raise ValueError('Pinned official CC0 opening anchors are required')
    missing=[name for name in ['torch','numpy','chess','zstandard','psutil'] if importlib.util.find_spec(name) is None]
    if missing:raise ValueError('Missing dependencies: '+', '.join(missing)+'; install requirements.txt in an isolated environment')
    return c

def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    args=parser().parse_args(arguments)
    trainer=None
    try:
        if args.resume:
            import sqlite3
            ledger=Path(args.data_directory)/'ledger.sqlite3'
            if ledger.is_file():
                with sqlite3.connect(ledger) as db:
                    row=db.execute("SELECT v FROM meta WHERE k='runtime_config'").fetchone() or db.execute("SELECT v FROM meta WHERE k='settings'").fetchone()
                if row:
                    previous=json.loads(row[0]);explicit={s.split('=',1)[0].lstrip('-').replace('-','_') for s in arguments if s.startswith('--')}
                    if 'no_keep_shards' in explicit:explicit.add('keep_shards')
                    for key in vars(args):
                        if key in previous and key not in explicit and key not in ['resume','dry_run']:setattr(args,key,previous[key])
        config=validate(args)
        if args.dry_run:
            print(json.dumps({'status':'dry-run-ok','configuration':config,'downloads_started':False,'training_started':False,'champion_will_be_overwritten':False},ensure_ascii=False,indent=2));return 0
        from .memory_limit import install
        limit=install(config['max_memory_gb'])
        from .storage import atomic_json,canonical
        from .calibration import Calibration
        from .anchors import opening_anchors
        from .streaming import SourceSpec
        from .pipeline import Trainer,RunLock
        calibration=Calibration.load(config['calibration'])
        anchors=opening_anchors(config['openings_directory'],min(2000,max(13,config['positions']//5)),config['seed'])
        excluded=set(calibration.excluded_families)
        puzzle_meta=ROOT/'fixtures/official-puzzles.csv.source.json'
        if puzzle_meta.exists():excluded.update(json.loads(puzzle_meta.read_text(encoding='utf-8')).get('excluded_training_families',[]))
        if not calibration.usable:print('Calibration did not pass independent validation: CP value targets will have zero weight; promotion is blocked.',flush=True)
        source=SourceSpec(config['source'],license='CC0-1.0',original_url=DEFAULT_SOURCE,version=config['source_provenance'].get('version','HTTP validators'))
        with RunLock(Path(config['data_directory'])/'trainer.lock'):
            trainer=Trainer(config,source,calibration,anchors,excluded)
            atomic_json(Path(config['checkpoint_directory'])/'resource-policy.json',limit)
            result=trainer.train();print(json.dumps({'status':'completed','report':result},ensure_ascii=False));return 0
    except KeyboardInterrupt:
        print('Interrupted. Resume uses the last committed checkpoint; unfinished shard retained.',file=sys.stderr);return 130
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        if trainer is not None:
            from .storage import atomic_json
            path=trainer.checkpoints/'status.json'
            old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            old.update(status='interrupted',last_error=f'{type(exc).__name__}: {exc}')
            atomic_json(path,old)
        return 1
    finally:
        if trainer is not None:trainer.close()

if __name__=='__main__':raise SystemExit(main())
