"""Manually register a quality-gated recommendation from completed pipeline rows."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from shardtrain.recommendation import load_rows,normalize_rows,build_recommendation,file_sha
from shardtrain.storage import atomic_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,nargs='+',required=True,help='Completed pipeline JSON or generic JSONL rows')
    p.add_argument('--metadata',type=Path,help='Optional benchmark JSON containing dataset/initial SHA and common settings')
    p.add_argument('--signature-file',type=Path,default=ROOT/'reports/benchmarks/microbenchmark.json')
    p.add_argument('--positions',type=int,default=10000);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--min-repeats',type=int,default=3);p.add_argument('--relative-loss-tolerance',type=float,default=.02)
    p.add_argument('--register',action='store_true',help='Explicitly register completed, passing evidence for auto-device matching')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    signature_document=json.loads(a.signature_file.read_text(encoding='utf-8'))
    signature=signature_document.get('signature',signature_document)
    raw,sources=load_rows(a.results,a.metadata);rows,artifacts=normalize_rows(raw,signature)
    sources+=artifacts+[{'path':str(a.signature_file.resolve()),'sha256':file_sha(a.signature_file)}]
    report=build_recommendation(rows,signature,sources,positions=a.positions,batch_size=a.batch_size,
        min_repeats=a.min_repeats,relative_loss_tolerance=a.relative_loss_tolerance,register=a.register)
    atomic_json(a.output,report)
    print(json.dumps({'status':report['status'],'registered':report['registered'],'device':report['device'],
        'cpu_threads':report['cpu_threads'],'batch_size':report['batch_size'],'full_pipeline_auto_profiles':len(report['auto_device_profiles']),
        'rows_total':len(rows),'excluded_rows':len(report['evidence']['excluded_rows']),'output':str(a.output)},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
