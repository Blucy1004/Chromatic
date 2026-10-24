"""Bounded, read-only full-vs-selected normalizer equivalence experiment.

No default pipeline change, training, GPU use, download, or all-corpus RAM load.
"""
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
import argparse,csv,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from shardtrain import quality as base
from shardtrain.quality_fast import normalize_selected,semantic_record
from shardtrain.calibration import Calibration
from shardtrain.storage import atomic_json,digest
from shardtrain.streaming import RecordStream,SourceSpec

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')

@contextmanager
def call_counters(counters,current):
    originals={name:getattr(base,name) for name in ['_pv','board_from_fen']}
    def wrapped(name):
        def invoke(*args,**kwargs):
            counters[current[0]][name]+=1
            return originals[name](*args,**kwargs)
        return invoke
    for name in originals:setattr(base,name,wrapped(name))
    try:yield
    finally:
        for name,value in originals.items():setattr(base,name,value)

def compare_records(records,*,calibration=None,config=None,max_records=60000,deadline_seconds=300,progress=None):
    if not 1<=max_records<=60000:raise ValueError('Record budget must be between 1 and 60000')
    if not 0<deadline_seconds<=900:raise ValueError('Experiment deadline must be at most 900 seconds')
    cfg=config or base.QualityConfig();started=time.perf_counter()
    timers=Counter();accepted=Counter();rejections={'full':Counter(),'selected':Counter()}
    calls={'full':Counter(),'selected':Counter()};current=['full'];digests={key:hashlib.sha256() for key in calls}
    mismatches=[];mismatch_count=0;processed=0;skipped=0;attempted=0;metadata_checked=0;stop='source_exhausted'
    with call_counters(calls,current):
        for line,raw in records:
            if processed>=max_records:stop='record_budget';break
            if time.perf_counter()-started>=deadline_seconds:stop='deadline';break
            source={'license':'CC0-1.0','corpus_line':line,'experiment':'full-vs-selected-equivalence'}
            values={};encoded_values={}
            for method in (['full','selected'] if processed%2==0 else ['selected','full']):
                current[0]=method;stamp=time.perf_counter()
                try:
                    result=base.parse_record(raw,calibration=calibration,config=cfg,source=source) if method=='full' else normalize_selected(raw,config=cfg,calibration=calibration,source=source)
                    outcome={'accepted':True,'semantic':semantic_record(result)};accepted[method]+=1
                    if method=='selected':
                        audit=result['source_eval'];skipped+=audit.get('skipped_lower_evaluations',0)
                        attempted+=len(audit.get('full_validation_attempted_indices',[]))
                        metadata_checked+=audit.get('metadata_checked_evaluations',0)
                except base.QualityError as exc:
                    outcome={'accepted':False,'error_type':'QualityError','reason':exc.reason};rejections[method][exc.reason]+=1
                except Exception as exc:
                    outcome={'accepted':False,'error_type':type(exc).__name__,'reason':str(exc)};rejections[method][type(exc).__name__]+=1
                timers[method]+=time.perf_counter()-stamp
                encoded=canonical(outcome);digests[method].update(len(encoded).to_bytes(8,'little'));digests[method].update(encoded)
                values[method]=outcome;encoded_values[method]=encoded
            if encoded_values['full']!=encoded_values['selected']:
                mismatch_count+=1
                if len(mismatches)<10:mismatches.append({'line':line,'full':values['full'],'selected':values['selected']})
            processed+=1
            if progress and processed%1000==0:progress({'processed':processed,'elapsed_seconds':time.perf_counter()-started,
                'normalization_seconds':dict(timers),'accepted':dict(accepted),'mismatch_count':mismatch_count})
    if processed==max_records:stop='record_budget'
    return {'format':'chromatic-quality-selected-validation-v1','records':processed,'max_records':max_records,
        'elapsed_seconds':time.perf_counter()-started,'stop_reason':stop,'requested_records_completed':processed==max_records,
        'normalization_seconds':dict(timers),'normalization_records_per_second':{method:processed/timers[method] if timers[method] else None for method in calls},
        'selected_speedup':timers['full']/timers['selected'] if timers['selected'] else None,
        'accepted':dict(accepted),'rejected':{k:dict(v) for k,v in rejections.items()},
        'semantic_outcome_sha256':{k:v.hexdigest() for k,v in digests.items()},'mismatch_count':mismatch_count,'mismatch_examples':mismatches,
        'calls':{k:dict(v) for k,v in calls.items()},'selected_audit_totals':{'metadata_checked':metadata_checked,
            'full_validation_attempted':attempted,'lower_analyses_skipped_uninspected':skipped},
        'semantic_equivalent':mismatch_count==0 and digests['full'].hexdigest()==digests['selected'].hexdigest(),
        'contract':'All board/history/identity/group/phase/policy/WDL/weights/selected-score/PV fields and provenance compare as canonical JSON bytes. Only explicitly changed rejection-audit scope is excluded.',
        'timing_contract':'Same parsed record, alternating method order. Normalizer timings include lightweight PV/FEN call counters, exclude source decode/JSON time. Bounded stream. No training.',
        'default_validation_mode_changed':False}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'fixtures/official-evaluations.jsonl.zst')
    p.add_argument('--calibration',type=Path,default=ROOT/'fixtures/calibration.json')
    p.add_argument('--output',type=Path,default=ROOT/'reports/quality-selected')
    p.add_argument('--max-records',type=int,default=60000);p.add_argument('--deadline-seconds',type=float,default=300)
    args=p.parse_args(argv);source=args.source.resolve()
    if not source.is_file():raise FileNotFoundError(source)
    sidecar=Path(str(source)+'.source.json')
    if not sidecar.is_file():raise ValueError('Local official source provenance required')
    provenance=json.loads(sidecar.read_text(encoding='utf-8'))
    if provenance.get('url')!='https://database.lichess.org/lichess_db_eval.jsonl.zst' or provenance.get('license')!='CC0-1.0' or provenance.get('local_sha256')!=digest(source):raise ValueError('Official CC0 source/hash mismatch')
    calibration=Calibration.load(args.calibration);args.output.mkdir(parents=True,exist_ok=True)
    before=digest(source);parse_seconds=0.;json_errors=0
    with RecordStream(SourceSpec(str(source),license='CC0-1.0',original_url=provenance['url'],version=provenance.get('version','local-official-fixture')),
                      spool_directory=args.output/'source-spool',max_compressed_bytes=16*1024**2,
                      max_decompressed_bytes=128*1024**2,max_frame_bytes=128*1024**2) as stream:
        def records():
            nonlocal parse_seconds,json_errors
            for line,text in stream.iter_lines():
                stamp=time.perf_counter()
                try:raw=json.loads(text)
                except json.JSONDecodeError:json_errors+=1;raw=None
                parse_seconds+=time.perf_counter()-stamp;yield line,raw
        def progress(row):atomic_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
        report=compare_records(records(),calibration=calibration,max_records=args.max_records,deadline_seconds=args.deadline_seconds,progress=progress)
        report.update(source=str(source),source_sha256=before,source_unchanged=digest(source)==before,
            calibration_sha256=digest(args.calibration),stream_stats=stream.stats,source_cursor=stream.cursor,
            json_parse_seconds=parse_seconds,json_parse_errors=json_errors)
    report['source_files_sha256']={name:digest(ROOT/name) for name in ['shardtrain/quality.py','shardtrain/quality_fast.py','tools/validate_quality_fast.py']}
    atomic_json(args.output/'validation.json',report)
    with (args.output/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['method','records','accepted','seconds','records_per_second','pv_calls','fen_calls','semantic_sha256'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for method in ['full','selected']:writer.writerow({'method':method,'records':report['records'],'accepted':report['accepted'].get(method,0),
            'seconds':report['normalization_seconds'].get(method,0),'records_per_second':report['normalization_records_per_second'][method],
            'pv_calls':report['calls'][method].get('_pv',0),'fen_calls':report['calls'][method].get('board_from_fen',0),
            'semantic_sha256':report['semantic_outcome_sha256'][method]})
    print(json.dumps({k:report[k] for k in ['records','semantic_equivalent','mismatch_count','selected_speedup','stop_reason','source_unchanged']}))
    return 0 if report['semantic_equivalent'] and report['source_unchanged'] else 1
if __name__=='__main__':raise SystemExit(main())
