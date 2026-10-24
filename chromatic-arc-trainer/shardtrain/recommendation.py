"""Quality-gated full-pipeline recommendations; never inferred from kernel speed.

Pure standard library: loading a recommendation does not initialize a GPU or
change a training configuration. Explicit devices must remain explicit.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

FORMAT='chromatic-arc-pipeline-recommendation-v1'
PROFILE_FIELDS=('positions','batch_size','cpu_threads','precision','cache_mode','prefetch',
                'optimizer','gradient_accumulation','transfer_mode','validation_mode')
BASE_FIELDS=('positions','source_sha256','initial_sha256','model_config','epochs','seed','replay_size',
             'shard_size','learning_rate','checkpoint_every','holdout_digest','source_cursor_line','quality_digest','validation_mode')
FAIR_FIELDS=('batch_size','global_step','training_samples','train_shards_sha256')
REQUIRED_RUNTIME=('positions','batch_size','cpu_threads','precision','cache_mode','prefetch')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def hash_value(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


def _implementation_signature():
    root=Path(__file__).resolve().parents[1]
    return {name:file_sha(root/name) for name in ['vendor/zero/model.py','vendor/zero/encoding.py']}


def valid_signature(signature):
    return isinstance(signature,dict) and isinstance(signature.get('details'),dict) and signature.get('key')==hash_value(signature['details'])


def _float(value):
    try:value=float(value)
    except (ValueError,TypeError):return None
    return value if math.isfinite(value) else None


def _integer(value):
    number=_float(value)
    return int(number) if number is not None and number==int(number) else None


def load_rows(paths,metadata_path=None):
    """Accept a JSON results list/object or line-delimited completed rows."""
    sources={};metadata={};rows=[]
    def read(path):
        path=Path(path).resolve();data=path.read_bytes()
        sources[str(path)]={'path':str(path),'sha256':hashlib.sha256(data).hexdigest()}
        return data.decode('utf-8')
    if metadata_path:
        obj=json.loads(read(metadata_path));metadata.update(obj.get('metadata',obj))
    for filename in paths:
        path=Path(filename).resolve();raw=read(path)
        try:obj=json.loads(raw)
        except json.JSONDecodeError:obj=None
        if isinstance(obj,dict) and isinstance(obj.get('results',obj.get('rows')),list):
            local=dict(metadata,**obj.get('metadata',{}));items=obj.get('results',obj.get('rows'))
        elif isinstance(obj,list):local=metadata;items=obj
        elif isinstance(obj,dict) and ('positions' in obj or 'row' in obj):local=metadata;items=[obj]
        else:
            local=dict(metadata);items=[]
            for line,body in enumerate(raw.splitlines(),1):
                if not body.strip():continue
                item=json.loads(body)
                if isinstance(item,dict) and item.get('type')=='metadata':local.update(item.get('metadata',{}))
                elif isinstance(item,dict):items.append(item)
                else:raise ValueError(f'JSONL row {line} must be an object')
        for number,item in enumerate(items,1):
            if not isinstance(item,dict):raise ValueError('Pipeline rows must be JSON objects')
            if isinstance(item.get('row'),dict):item=dict(item['row'])
            rows.append((dict(item),dict(local),str(path),number))
    return rows,list(sources.values())


def normalize_rows(raw_rows,signature):
    """Hydrate saved CLI/manifest evidence read-only; never execute argv."""
    sources={};json_cache={};sha_cache={};normalized=[]
    def read_json(path):
        path=Path(path).resolve()
        if not path.is_file():return None
        key=str(path)
        if key not in json_cache:
            data=path.read_bytes();json_cache[key]=json.loads(data.decode('utf-8'))
            sources[key]={'path':key,'sha256':hashlib.sha256(data).hexdigest()}
        return json_cache[key]
    def actual_sha(path):
        key=str(Path(path).resolve())
        if key not in sha_cache:sha_cache[key]=file_sha(key)
        return sha_cache[key]
    for number,(row,metadata,source_file,line) in enumerate(raw_rows):
        reasons=[];command={};report={};manifest={}
        command_path=row.get('command')
        try:
            if isinstance(command_path,str):command=read_json(command_path) or {}
            elif isinstance(command_path,dict):command=command_path
            directory=Path(row['directory']) if row.get('directory') else None
            if isinstance(row.get('training_report'),dict):report=row['training_report']
            elif directory is not None:report=read_json(directory/'candidate/training-report.json') or {}
            manifest_path=report.get('candidate_manifest')
            if manifest_path:manifest=read_json(manifest_path) or {}
        except (OSError,ValueError,TypeError) as exc:reasons.append('unreadable_artifact:'+type(exc).__name__)
        options={};argv=command.get('argv',[])
        for index,token in enumerate(argv):
            if isinstance(token,str) and token.startswith('--'):
                key=token[2:].replace('-','_');options[key]=argv[index+1] if index+1<len(argv) and not str(argv[index+1]).startswith('--') else True
        settings=dict(metadata.get('settings',{}),**row.get('settings',{}))
        def get(key,default=None,aliases=()):
            for obj in [row,options,settings]:
                for name in [key,*aliases]:
                    if obj.get(name) is not None:return obj[name]
            return default
        expected_config=signature['details'].get('model_config')
        config=row.get('model_config') or manifest.get('config') or metadata.get('model_config')
        initial=row.get('initial_sha256') or metadata.get('champion_sha256') or metadata.get('initial_sha256')
        source_sha=row.get('source_sha256') or metadata.get('source_sha256')
        # A command's actual input is stronger evidence than a row assertion.
        for key,expected in [('source',source_sha),('initial_checkpoint',initial)]:
            path=options.get(key)
            if path and Path(path).is_file():
                measured=actual_sha(path)
                if expected and measured!=expected:reasons.append(key+'_sha256_mismatch')
                elif key=='source':source_sha=measured
                else:initial=measured
        if config!=expected_config:reasons.append('model_config_not_measured_by_signature')
        measured_signature=row.get('signature') or metadata.get('signature') or metadata.get('hardware_signature')
        measured_key=row.get('signature_key') or metadata.get('signature_key')
        if measured_signature is None and measured_key is None:reasons.append('missing_measured_hardware_signature')
        if measured_signature is not None and measured_signature!=signature:reasons.append('stale_hardware_or_model_signature')
        if measured_key is not None and measured_key!=signature['key']:reasons.append('stale_hardware_or_model_signature')
        source_cursor=report.get('source_cursor',{})
        quality=row.get('quality_stats') or report.get('quality')
        train_sha=row.get('train_shards_sha256') or row.get('training_identity_sha256')
        if train_sha is None and row.get('directory'):
            database=Path(row['directory'])/'data/ledger.sqlite3'
            if database.is_file():
                try:
                    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
                        shards=db.execute('SELECT id,sha,rows,status FROM shards ORDER BY id').fetchall()
                    if shards and all(item[3]=='done' for item in shards):train_sha=hash_value([item[:3] for item in shards])
                    else:reasons.append('shards_not_all_completed')
                except sqlite3.Error:reasons.append('unreadable_shard_identity_evidence')
        execution=report.get('execution',{});device_info=report.get('device',{})
        positions=_integer(get('positions'));processed=_integer(row.get('processed_positions',report.get('positions')))
        device=get('device');resolved=row.get('resolved_device',device_info.get('device',device))
        precision=get('precision','fp32');effective=row.get('effective_precision',execution.get('effective_precision',device_info.get('precision',precision)))
        precision_fallback=row.get('precision_fallbacks',execution.get('precision_fallbacks'))
        profile=bool(get('profile',False)) or bool(row.get('profile_enabled',False))
        n={'row_id':str(row.get('key',f'{row.get("name",source_file)}:{row.get("repeat",line)}')),
            'source_file':source_file,'source_line':line,'implementation':get('implementation','arc'),
            'status':row.get('status'),'returncode':row.get('returncode'),
            'device':device,'resolved_device':resolved,'precision':precision,'effective_precision':effective,
            'positions':positions,'processed_positions':processed,'batch_size':_integer(get('batch_size')),
            'cpu_threads':_integer(get('cpu_threads',aliases=('threads',))),'cache_mode':get('cache_mode','auto'),
            'prefetch':_integer(get('prefetch',0)),'optimizer':get('optimizer','default'),
            'gradient_accumulation':_integer(get('gradient_accumulation',1,('accumulation',))),
            'transfer_mode':get('transfer_mode','copy'),'validation_mode':get('validation_mode','full'),
            'epochs':_integer(get('epochs')),'seed':_integer(get('seed')),'replay_size':_integer(get('replay_size')),
            'shard_size':_integer(get('shard_size')),'learning_rate':_float(get('learning_rate')),
            'checkpoint_every':_integer(get('checkpoint_every')),'source_sha256':source_sha,'initial_sha256':initial,
            'model_config':config,'holdout_digest':row.get('holdout_digest',report.get('holdout_digest')),
            'source_cursor_line':_integer(row.get('source_cursor_line',source_cursor.get('line'))),
            'quality_digest':row.get('quality_digest') or (hash_value(quality) if quality is not None else None),
            'train_shards_sha256':train_sha,'global_step':_integer(row.get('global_step',report.get('global_step'))),
            'training_samples':_integer(row.get('training_samples',execution.get('training_samples'))),
            'wall_seconds':_float(row.get('wall_seconds')),
            'policy_validation_loss':_float(row.get('policy_validation_loss',report.get('validation',{}).get('policy_loss'))),
            'wdl_validation_loss':_float(row.get('wdl_validation_loss',report.get('validation',{}).get('wdl_loss'))),
            'repeat':_integer(row.get('repeat',1)),'profile':profile,'reasons':reasons}
        if n['status'] not in ['ok','completed'] or n['returncode']!=0:reasons.append('run_not_successfully_completed')
        if positions is None or positions<=0 or positions!=processed:reasons.append('processed_position_count_incomplete')
        if profile:reasons.append('profile_run_not_eligible_for_timing_selection')
        if resolved!=device:reasons.append('device_fallback_not_a_device_measurement')
        if effective!=precision or precision_fallback:reasons.append('precision_fallback_not_a_precision_measurement')
        if device not in ['cpu','xpu']:reasons.append('unsupported_device')
        if n['wall_seconds'] is None or n['wall_seconds']<=0:reasons.append('nonfinite_or_nonpositive_wall_time')
        for key in ['policy_validation_loss','wdl_validation_loss']:
            if n[key] is None or n[key]<0:reasons.append('nonfinite_or_missing_'+key)
        for key in BASE_FIELDS+FAIR_FIELDS:
            if n.get(key) is None:reasons.append('missing_comparability_evidence:'+key)
        for key in ['batch_size','cpu_threads','epochs','shard_size','checkpoint_every','global_step','training_samples','gradient_accumulation','repeat']:
            if n.get(key) is not None and n[key]<=0:reasons.append('nonpositive_'+key)
        if n['prefetch'] is not None and n['prefetch']<0:reasons.append('negative_prefetch')
        if n['replay_size'] is not None and n['replay_size']<0:reasons.append('negative_replay_size')
        if n['learning_rate'] is not None and n['learning_rate']<=0:reasons.append('nonpositive_learning_rate')
        normalized.append(n)
    return normalized,list(sources.values())


def _key(row,fields):return canonical({name:row.get(name) for name in fields})


def _summary(rows):
    walls=[row['wall_seconds'] for row in rows];mean=statistics.mean(walls)
    return {'repeats':len(rows),'wall_seconds_median':statistics.median(walls),'wall_seconds_min':min(walls),
        'wall_seconds_max':max(walls),'wall_seconds_cv':statistics.pstdev(walls)/mean if len(walls)>1 else 0.,
        'policy_validation_loss_median':statistics.median(row['policy_validation_loss'] for row in rows),
        'wdl_validation_loss_median':statistics.median(row['wdl_validation_loss'] for row in rows),
        'row_ids':[row['row_id'] for row in rows]}


def build_recommendation(rows,signature,sources=(),*,positions=10000,batch_size=32,min_repeats=3,relative_loss_tolerance=.02,register=False):
    if not valid_signature(signature):raise ValueError('Invalid hardware/model signature hash')
    if not 0<=relative_loss_tolerance<=.02:raise ValueError('Loss tolerance must be predeclared between 0 and 2%')
    if positions<1 or batch_size<1 or min_repeats<1:raise ValueError('Positive positions, batch size and repeat count required')
    rows=[dict(row,reasons=list(row.get('reasons',[]))) for row in rows]
    seen=set()
    for row in rows:
        if row['row_id'] in seen:row['reasons'].append('duplicate_completed_row_not_an_independent_repeat')
        seen.add(row['row_id'])
    cohorts=defaultdict(list)
    for row in rows:
        if not row['reasons']:cohorts[_key(row,BASE_FIELDS)].append(row)
    references={};qualified=[]
    for base,cohort in cohorts.items():
        cpu=[row for row in cohort if row['device']=='cpu' and row['precision']=='fp32' and row['cpu_threads']==1 and row['batch_size']==batch_size]
        preferred=[row for row in cpu if row['implementation']=='arc'];cpu=preferred or cpu
        if not cpu:
            for row in cohort:row['reasons'].append('no_matching_cpu1_fp32_quality_reference')
            continue
        # A reference may not mix different cache/prefetch/optimizer regimes.
        by_condition=defaultdict(list)
        for row in cpu:by_condition[_key(row,PROFILE_FIELDS[1:])].append(row)
        reference_rows=max(by_condition.values(),key=lambda group:(len(group),-statistics.median(r['wall_seconds'] for r in group)))
        if len(reference_rows)<min_repeats:
            for row in cohort:row['reasons'].append('cpu_quality_reference_has_insufficient_successful_repeats')
            continue
        reference=reference_rows[0];losses=_summary(reference_rows)
        references[base]={'constraints':{key:reference[key] for key in BASE_FIELDS+FAIR_FIELDS},'measurement':losses}
        for row in cohort:
            for metric in ['policy_validation_loss','wdl_validation_loss']:
                limit=losses[metric+'_median']*(1+relative_loss_tolerance)+1e-8
                if row[metric]>limit:row['reasons'].append('quality_gate_'+metric+'_worse_than_cpu_reference')
            for key in FAIR_FIELDS:
                if row[key]!=reference[key]:row['reasons'].append('fair_comparison_'+key+'_differs_from_cpu_reference')
            if row['implementation']!='arc':row['reasons'].append('historical_implementation_not_an_arc_candidate')
            if not row['reasons']:qualified.append(row)
    # Profile key includes data, exact updates, cache/prefetch and runtime knobs.
    grouped=defaultdict(list)
    for row in qualified:grouped[_key(row,BASE_FIELDS+FAIR_FIELDS+PROFILE_FIELDS+('device',))].append(row)
    candidates=[]
    for group in grouped.values():
        reference=group[0];measurement=_summary(group);reasons=[]
        if len({row['repeat'] for row in group})!=len(group):reasons.append('duplicate_repeat_number')
        if len(group)<min_repeats:reasons.append('insufficient_successful_repeats')
        if reference['positions']!=positions:reasons.append('different_measured_position_scale')
        candidates.append({'device':reference['device'],'constraints':{key:reference[key] for key in dict.fromkeys(BASE_FIELDS+FAIR_FIELDS+PROFILE_FIELDS)},
            'measurement':measurement,'eligible':not reasons,'reasons':reasons})
    profiles=[];profile_groups=defaultdict(list)
    for candidate in candidates:
        if candidate['eligible']:profile_groups[canonical(candidate['constraints'])].append(candidate)
    for constraints,group in profile_groups.items():
        by_device={row['device']:row for row in group}
        if 'cpu' not in by_device or 'xpu' not in by_device:continue
        chosen=min(group,key=lambda row:(row['measurement']['wall_seconds_median'],row['device']))
        profiles.append({'constraints':json.loads(constraints),'device':chosen['device'],
            'measurement':chosen['measurement'],'device_measurements':{name:value['measurement'] for name,value in by_device.items()},
            'both_devices_measured':True,'quality_gate_passed':True})
    eligible_cpu=[c for c in candidates if c['eligible'] and c['device']=='cpu' and c['constraints']['precision']=='fp32']
    best_cpu=min(eligible_cpu,key=lambda row:row['measurement']['wall_seconds_median']) if eligible_cpu else None
    best_profile=min(profiles,key=lambda row:row['measurement']['wall_seconds_median']) if profiles else None
    selection=best_cpu or best_profile
    report={'format':FORMAT,'created_utc':datetime.now(timezone.utc).isoformat(),'registered':bool(register and selection),
        'status':'registered' if register and selection else 'review' if selection else 'insufficient_evidence',
        'signature':signature,'model_config':signature['details'].get('model_config'),'model_size_specific':True,
        'model_implementation_sha256':_implementation_signature(),
        'device':selection['device'] if selection else None,
        'batch_size':selection['constraints']['batch_size'] if selection else None,
        'cpu_threads':selection['constraints']['cpu_threads'] if selection else None,
        'constraints':selection['constraints'] if selection else {},'measurement':selection['measurement'] if selection else {},
        'scope':'Measured full pipeline at the exact corpus/initial model/scale/epochs/seed/replay/update count; quality-gated CPU thread recommendation. Auto-device matching uses only profiles with both CPU and XPU measured under the same runtime constraints.',
        'quality_gate':{'relative_loss_tolerance':relative_loss_tolerance,'absolute_loss_epsilon':1e-8,
            'policy_and_wdl_individually_required':True,'finite_completed_only':True,'minimum_repeats':min_repeats},
        'best_cpu':best_cpu,'auto_device_profiles':profiles,'candidate_groups':candidates,
        'evidence':{'sources':list({source['path']:source for source in sources}.values()),'rows':rows,
            'cpu_quality_references':list(references.values()),'positions_requested':positions,'batch_size_reference':batch_size,
            'rows_total':len(rows),'rows_quality_and_comparability_passed':len(qualified),
            'excluded_rows':[{'row_id':r['row_id'],'positions':r['positions'],'batch_size':r['batch_size'],'reasons':r['reasons']} for r in rows if r['reasons']]},
        'not_proven':['No chess-strength promotion or Elo claim.','Microbenchmark batch 256/512 results do not establish matching epoch/update quality.','No extrapolation to another model architecture, driver, hardware, torch version, corpus or position count.']}
    report['artifact_sha256']=hash_value(report);return report


def read_validated(path,signature,constraints):
    """Return an exact matching full-pipeline auto profile, otherwise None.

    Mandatory runtime keys are positions/batch_size/cpu_threads/precision/
    cache_mode/prefetch. Additional supplied training constraints are all checked.
    Source files must still match their registered hashes. This function does not
    apply settings and never overrides an explicit cpu/xpu request.
    """
    try:
        if constraints.get('device') not in [None,'auto']:return None
        if not valid_signature(signature) or any(key not in constraints for key in REQUIRED_RUNTIME):return None
        report=json.loads(Path(path).read_text(encoding='utf-8'))
        if report.get('format')!=FORMAT or not report.get('registered') or report.get('status')!='registered':return None
        body=dict(report);expected=body.pop('artifact_sha256',None)
        if expected!=hash_value(body) or report.get('signature')!=signature:return None
        if report.get('model_config')!=signature['details'].get('model_config'):return None
        if report.get('model_implementation_sha256')!=_implementation_signature():return None
        if not report.get('quality_gate',{}).get('finite_completed_only') or report['quality_gate'].get('relative_loss_tolerance',1)>.02:return None
        for source in report.get('evidence',{}).get('sources',[]):
            if not Path(source['path']).is_file() or file_sha(source['path'])!=source['sha256']:return None
        options=[]
        for profile in report.get('auto_device_profiles',[]):
            if not profile.get('both_devices_measured') or not profile.get('quality_gate_passed'):continue
            actual=profile.get('constraints',{})
            if all(actual.get(key)==value for key,value in constraints.items() if key!='device'):options.append(profile)
        if not options:return None
        selected=min(options,key=lambda item:item['measurement']['wall_seconds_median'])
        return {'format':FORMAT,'signature':signature,'registered':True,'device':selected['device'],
            'constraints':selected['constraints'],'measurement':selected['measurement'],
            'batch_size':selected['constraints']['batch_size'],'cpu_threads':selected['constraints']['cpu_threads'],
            'precision':selected['constraints']['precision'],'model_config':report['model_config'],'scope':report['scope'],
            'evidence_path':str(Path(path).resolve()),'artifact_sha256':expected}
    except (OSError,ValueError,TypeError,KeyError):return None
