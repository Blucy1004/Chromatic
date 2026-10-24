"""Read-only data quality, leakage and recovery manifest audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

def audit(data_directory):
    directory=Path(data_directory);uri=(directory/'ledger.sqlite3').resolve().as_uri()+'?mode=ro'
    with sqlite3.connect(uri,uri=True) as db:
        def meta(key):
            row=db.execute('SELECT v FROM meta WHERE k=?',(key,)).fetchone();return json.loads(row[0]) if row else None
        families={s:{r[0] for r in db.execute('SELECT family FROM seen_families WHERE split=?',(s,))} for s in ['train','validation','test']}
        overlaps={a+'/'+b:len(families[a]&families[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
        fixed=meta('fixed_sets');shards=[{'id':i,'rows':n,'status':s,'file_exists':Path(p).is_file()} for i,n,s,p in db.execute('SELECT id,rows,status,path FROM shards ORDER BY id')]
        result={'quality_counters':meta('quality_stats'),'source_cursor':meta('source_cursor'),
                'fixed_sets':fixed,'family_overlap':overlaps,'split_family_counts':{k:len(v) for k,v in families.items()},
                'shards':shards,'anchor_catalog':meta('anchors_initialized'),'all_split_groups_disjoint':not any(overlaps.values()),
                'policy_note':'Single/multi PV labels cover observed support only; weak confidence-weighted imitation.',
                'value_note':'Missing CP calibration or opening value labels have value weight zero.'}
        if not result['all_split_groups_disjoint']:raise RuntimeError('Train/validation/test family leakage')
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-directory',required=True,type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args(argv);result=audit(args.data_directory)
    if args.output:
        from .storage import atomic_json
        atomic_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
