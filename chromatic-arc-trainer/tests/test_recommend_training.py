"""Recommendations require full-pipeline, comparable, finite quality evidence."""
import copy
import json
from pathlib import Path

import pytest

from shardtrain.recommendation import (FORMAT,hash_value,load_rows,normalize_rows,
    build_recommendation,read_validated,file_sha)


def signature(width=64,hardware='Arc 140T'):
    details={'schema':1,'model_config':{'width':width,'blocks':2},'torch':'2.14.1+xpu','xpu':[hardware]}
    return {'key':hash_value(details),'details':details}


def row(device='cpu',threads=1,repeat=1,wall=10.,positions=10000,batch=32,**extra):
    result={'key':f'{device}-{threads}-{repeat}-{positions}-{batch}','status':'ok','returncode':0,
        'implementation':'arc','device':device,'resolved_device':device,'precision':'fp32','effective_precision':'fp32',
        'positions':positions,'processed_positions':positions,'batch_size':batch,'cpu_threads':threads,
        'cache_mode':'auto','prefetch':0,'optimizer':'default','gradient_accumulation':1,'transfer_mode':'copy',
        'validation_mode':'full','epochs':3,'seed':2026,'replay_size':256,'shard_size':1000 if positions==10000 else 250,
        'learning_rate':.001,'checkpoint_every':32,'source_sha256':'a'*64,'initial_sha256':'b'*64,
        'model_config':{'width':64,'blocks':2},'signature_key':signature()['key'],'holdout_digest':('c' if positions==10000 else 'd')*64,
        'source_cursor_line':25728 if positions==10000 else 1536,'quality_digest':'e'*64,'train_shards_sha256':'f'*64,
        'global_step':1149 if positions==10000 else 114,'training_samples':36318 if positions==10000 else 3618,
        'wall_seconds':wall,'policy_validation_loss':2.883 if positions==10000 else 3.065,
        'wdl_validation_loss':.724 if positions==10000 else .788,'repeat':repeat,'profile':False}
    result.update(extra);return result


def normalized(rows,sig=None):
    return normalize_rows([(r,{},'synthetic-completed.jsonl',i+1) for i,r in enumerate(rows)],sig or signature())[0]


def fair_rows():
    return ([row('cpu',1,r,wall=w) for r,w in enumerate([10.,11.,12.],1)]+
        [row('cpu',2,r,wall=w) for r,w in enumerate([8.,7.,9.],1)]+
        [row('xpu',1,r,wall=w,policy_validation_loss=2.884,wdl_validation_loss=.7241) for r,w in enumerate([14.,15.,16.],1)])


def runtime_constraints(threads=1):
    return {'positions':10000,'batch_size':32,'cpu_threads':threads,'precision':'fp32','cache_mode':'auto','prefetch':0}


def save(path,report):path.write_text(json.dumps(report),encoding='utf-8')


def test_three_repeats_choose_cpu_threads_and_only_matching_device_profile(tmp_path):
    report=build_recommendation(normalized(fair_rows()),signature(),register=True)
    assert report['format']==FORMAT and report['registered']
    assert report['device']=='cpu' and report['cpu_threads']==2
    assert report['measurement']['wall_seconds_median']==8
    assert report['measurement']['repeats']==3
    assert len(report['auto_device_profiles'])==1
    path=tmp_path/'recommendation.json';save(path,report)
    chosen=read_validated(path,signature(),runtime_constraints())
    assert chosen['device']=='cpu' and chosen['cpu_threads']==1
    assert chosen['measurement']['wall_seconds_median']==11
    assert read_validated(path,signature(),runtime_constraints(2)) is None


def test_registration_is_explicit_and_incomplete_runtime_keys_rejected(tmp_path):
    path=tmp_path/'r.json';report=build_recommendation(normalized(fair_rows()),signature())
    save(path,report);assert report['status']=='review' and not report['registered']
    assert read_validated(path,signature(),runtime_constraints()) is None
    report=build_recommendation(normalized(fair_rows()),signature(),register=True);save(path,report)
    constraints=runtime_constraints();del constraints['positions']
    assert read_validated(path,signature(),constraints) is None
    assert read_validated(path,signature(),dict(runtime_constraints(),device='xpu')) is None


def test_stale_hardware_and_another_model_architecture_are_rejected(tmp_path):
    report=build_recommendation(normalized(fair_rows()),signature(),register=True);path=tmp_path/'r.json';save(path,report)
    assert read_validated(path,signature(hardware='Different GPU'),runtime_constraints()) is None
    assert read_validated(path,signature(width=128),runtime_constraints()) is None
    changed=signature();changed['key']='0'*64
    assert read_validated(path,changed,runtime_constraints()) is None
    with pytest.raises(ValueError,match='signature'):build_recommendation(normalized(fair_rows()),changed)


def test_model_implementation_change_invalidates_same_width_configuration(tmp_path,monkeypatch):
    import shardtrain.recommendation as recommendation
    report=build_recommendation(normalized(fair_rows()),signature(),register=True);path=tmp_path/'r.json';save(path,report)
    changed=dict(report['model_implementation_sha256']);changed['vendor/zero/model.py']='0'*64
    monkeypatch.setattr(recommendation,'_implementation_signature',lambda:changed)
    assert read_validated(path,signature(),runtime_constraints()) is None


def test_batch256_and_512_quality_drift_is_not_a_faster_recommendation():
    rows=[row('cpu',1,r,wall=10,positions=1000) for r in range(1,4)]
    for batch,loss,steps in [(256,.869,15),(512,.975,12)]:
        rows.extend(row('cpu',8,r,wall=1,positions=1000,batch=batch,wdl_validation_loss=loss,
            global_step=steps,training_samples=3300) for r in range(1,4))
    report=build_recommendation(normalized(rows),signature(),positions=1000,register=True)
    assert report['batch_size']==32 and report['cpu_threads']==1
    excluded=report['evidence']['excluded_rows'];assert len(excluded)==6
    for item in excluded:
        assert 'quality_gate_wdl_validation_loss_worse_than_cpu_reference' in item['reasons']
        assert 'fair_comparison_global_step_differs_from_cpu_reference' in item['reasons']
        assert 'fair_comparison_batch_size_differs_from_cpu_reference' in item['reasons']
    large_scale=build_recommendation(normalized(rows),signature(),positions=10000,register=True)
    assert not large_scale['registered'] and large_scale['status']=='insufficient_evidence'


def test_failed_timed_out_nonfinite_and_profile_rows_are_excluded():
    rows=fair_rows()
    rows.extend([row('xpu',1,4,wall=.01,status='timeout',returncode=124),
        row('xpu',1,5,wall=.01,wdl_validation_loss=float('nan')),
        row('xpu',1,6,wall=float('inf')),
        row('xpu',1,7,wall=.01,profile=True),
        row('xpu',1,8,wall=.01,processed_positions=9999),
        row('xpu',1,9,wall=.01,effective_precision='bf16')])
    report=build_recommendation(normalized(rows),signature(),register=True)
    assert len(report['evidence']['excluded_rows'])==6
    assert report['auto_device_profiles'][0]['device']=='cpu'
    assert report['auto_device_profiles'][0]['device_measurements']['xpu']['wall_seconds_median']==15


def test_optimizer_updates_and_processed_corpus_must_match_reference():
    rows=[row('cpu',1,r) for r in range(1,4)]
    rows.extend(row('xpu',1,r,wall=1,global_step=1148) for r in range(1,4))
    report=build_recommendation(normalized(rows),signature(),register=True)
    assert not report['auto_device_profiles']
    assert all('fair_comparison_global_step_differs_from_cpu_reference' in x['reasons'] for x in report['evidence']['excluded_rows'])
    rows=[row('cpu',1,r) for r in range(1,4)]+[row('xpu',1,r,wall=1,train_shards_sha256='9'*64) for r in range(1,4)]
    report=build_recommendation(normalized(rows),signature(),register=True)
    assert not report['auto_device_profiles']


def test_validation_mode_cache_transfer_optimizer_and_prefetch_are_not_mixed():
    base=[row('cpu',1,r) for r in range(1,4)]
    for changed in [{'optimizer':'fused'},{'cache_mode':'device'},{'prefetch':2},{'transfer_mode':'preallocated'},
                    {'validation_mode':'selected'},{'gradient_accumulation':2},{'precision':'bf16','effective_precision':'bf16'}]:
        candidate=[row('xpu',1,r,wall=1,**changed) for r in range(1,4)]
        report=build_recommendation(normalized(base+candidate),signature(),register=True)
        assert not report['auto_device_profiles'],changed


def test_insufficient_or_duplicate_repeats_do_not_establish_medians():
    rows=[row('cpu',1,1),row('cpu',1,2),row('xpu',1,1),row('xpu',1,2)]
    report=build_recommendation(normalized(rows),signature(),register=True)
    assert not report['registered']
    duplicate=[row('cpu',1,1)]*3+[row('xpu',1,1)]*3
    report=build_recommendation(normalized(duplicate),signature(),register=True)
    assert not report['registered']


def test_source_hash_changes_and_artifact_corruption_invalidate_registration(tmp_path):
    source=tmp_path/'rows.jsonl';source.write_text('\n'.join(json.dumps(row) for row in fair_rows()),encoding='utf-8')
    raw,sources=load_rows([source]);rows,_=normalize_rows(raw,signature())
    report=build_recommendation(rows,signature(),sources,register=True);path=tmp_path/'r.json';save(path,report)
    assert read_validated(path,signature(),runtime_constraints())
    source.write_text(source.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    assert read_validated(path,signature(),runtime_constraints()) is None
    report['evidence']['sources']=[];report['device']='xpu';save(path,report)
    assert read_validated(path,signature(),runtime_constraints()) is None


def test_generic_jsonl_plus_metadata_and_saved_command_are_supported(tmp_path):
    source=tmp_path/'input.zst';source.write_bytes(b'trusted local fixture')
    meta={'metadata':{'source_sha256':file_sha(source),'champion_sha256':'b'*64,
        'settings':{'epochs':3,'seed':2026,'replay_size':256,'checkpoint_every':32}}}
    metadata=tmp_path/'metadata.json';metadata.write_text(json.dumps(meta),encoding='utf-8')
    rows=fair_rows()
    for item in rows:
        item['source_sha256']=file_sha(source)
        item['command']={'argv':['python','-m','shardtrain.run','--source',str(source),'--shard-size','1000','--learning-rate','.001']}
        for key in ['epochs','seed','replay_size','checkpoint_every','shard_size','learning_rate']:item.pop(key)
    path=tmp_path/'completed.jsonl';path.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
    raw,sources=load_rows([path],metadata);normalized_rows,artifacts=normalize_rows(raw,signature())
    report=build_recommendation(normalized_rows,signature(),sources+artifacts,register=True)
    assert report['registered'] and len(report['auto_device_profiles'])==1
    normalized_rows[0]['source_sha256']='0'*64
    again=build_recommendation(normalized_rows,signature(),register=True)
    assert any('cpu_quality_reference_has_insufficient_successful_repeats' in item['reasons'] for item in again['evidence']['excluded_rows'])
    assert not again['auto_device_profiles']
