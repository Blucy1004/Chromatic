#!/usr/bin/env python3
"""Read-only link from canonical candidates to preserved training bundles."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    index=ROOT/'chromatic-shard-trainer/candidates/index.json';preflight=read(ROOT/'project-chromatic-overlord/reports/factorial/preflight/models-and-resume.json')['models']
    result={'candidate_index_path':str(index),'candidate_index_sha256':sha(index),'candidates':{}}
    for n in [100,1000,10000]:
        model=f'lichess{n}';e=read(index)['candidates'][str(n)];directory=ROOT/f'chromatic-shard-validation/positions-{n}/candidate'
        report_path=directory/'training-report.json';report=read(report_path)
        name=e['source_manifest'].replace('\\','/').split('/')[-1];manifest_path=directory/'checkpoints'/name;manifest=read(manifest_path)
        assert report['positions']==n and report['global_step']==e['additional_updates']==preflight[model]['updates']==manifest['step']
        assert report['candidate_manifest'].replace('\\','/').split('/')[-1]==name
        files={}
        for kind in ['checkpoint','mirror','model']:
            f=manifest['files'][kind];path=manifest_path.parent/f['path'].replace('\\','/').split('/')[-1]
            actual=sha(path);assert path.stat().st_size==f['bytes'] and actual==f['sha256']
            canonical=Path(preflight[model]['model_path' if kind=='model' else 'pt_path'])
            assert actual==sha(canonical)
            files[kind]={'actual_path':str(path),'sha256':actual,'canonical_equal':str(canonical)}
        assert preflight[model]['initial_checkpoint_sha256']==preflight['zero128']['pt_sha256']
        result['candidates'][model]={'training_report_path':str(report_path),'training_report_sha256':sha(report_path),
            'bundle_manifest_path':str(manifest_path),'bundle_manifest_sha256':sha(manifest_path),'bundle_id':manifest['id'],'commit_seq':manifest['commit_seq'],
            'positions_processed':n,'additional_updates':manifest['step'],'completed_shards':manifest['completed'],
            'initial_pt_sha256':preflight[model]['initial_checkpoint_sha256'],'architecture':manifest['config'],'files':files,
            'source_cursor':report['source_cursor'],'raw_lines':report['quality']['raw_lines'],'training_history_reexecution':False}
    source=ROOT/'chromatic-shard-trainer/fixtures/official-evaluations.jsonl.zst'
    result['current_local_training_input']={'path':str(source),'sha256':sha(source),'bytes':source.stat().st_size,
        'scope':'preserved official eval local fixture; not a hash of the complete upstream database'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print('three canonical candidates match bundle/model/PT/mirror and training history')
if __name__=='__main__':main()
