"""Inspect or repair this tool's own committed checkpoint chain; never train."""
import argparse
import json
from pathlib import Path
import sqlite3

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-directory',type=Path,required=True);parser.add_argument('--checkpoint-directory',type=Path,required=True)
    parser.add_argument('--repair',action='store_true',help='Restore the authoritative head to a verified mirror/older commit')
    parser.add_argument('--max-memory-gb',type=float,default=2)
    args=parser.parse_args(argv)
    from .memory_limit import install
    install(args.max_memory_gb)
    from .durable import load_bundle,verify_bundle
    from .storage import Ledger,atomic_json
    from .pipeline import RunLock
    if args.repair:
        with RunLock(args.data_directory/'trainer.lock'):
            ledger=Ledger(args.data_directory/'ledger.sqlite3')
            try:
                state,path,recovered=load_bundle(ledger,args.checkpoint_directory)
                if state is None:raise RuntimeError('No committed model checkpoint; resume initial ingestion normally')
                result={'status':'recovery-ready','global_step':state['step'],'progress':state['progress'],
                        'checkpoint':str(path),'recovered':recovered,'training_started':False,'champion_modified':False}
                ledger.export_logs(args.checkpoint_directory/'logs');atomic_json(args.checkpoint_directory/'recovery-report.json',result)
            finally:ledger.close()
    else:
        uri=(args.data_directory/'ledger.sqlite3').resolve().as_uri()+'?mode=ro'
        with sqlite3.connect(uri,uri=True) as db:row=db.execute("SELECT v FROM meta WHERE k='current_manifest'").fetchone()
        if not row:raise RuntimeError('No committed checkpoint')
        path=json.loads(row[0]);state,mirror=verify_bundle(path)
        result={'status':'verified','global_step':state['step'],'progress':state['progress'],
                'checkpoint':path,'mirror_needed':mirror,'training_started':False,'champion_modified':False}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
