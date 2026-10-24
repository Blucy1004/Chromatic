"""Read-only candidate evaluation; no model is ever promoted or overwritten."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from .storage import atomic_json,canonical,digest
from .model_adapter import PolicyWDL,ModelConfig,checkpoint,features,action,measurements,torch,np
from .calibration import Calibration,RESULT_CLASS
from .quality import family_for
from .durable import read_manifest,verify_bundle
import chess
from zero.native import Native
from zero.match import arena

OPTIONS={'CPuct':'1.5','FPUReduction':'0','TreeMemoryMB':64,'MaxNodes':1000000,
         'MateProofNodes':0,'InferenceBackend':'SIMD','SparsePolicyHead':True,'SearchTelemetry':False}


def _connect(data_directory):
    path=(Path(data_directory)/'ledger.sqlite3').resolve()
    if not path.is_file():raise ValueError('run ledger does not exist')
    return sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=15)


def _meta(db,key,default=None):
    row=db.execute('SELECT v FROM meta WHERE k=?',(key,)).fetchone()
    return json.loads(row[0]) if row else default


def _load_model(state):
    # Loading an evaluation model must not advance a live trainer's RNG.
    with torch.random.fork_rng(devices=[]):
        model=PolicyWDL(ModelConfig(**state['config']))
        model.load_state_dict(state['model']);model.eval()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('nonfinite evaluation model')
    return model


def _export_matches(model,expected,directory,label):
    fd,name=tempfile.mkstemp(prefix=label+'-export-check-',suffix='.zero',dir=directory)
    os.close(fd);target=Path(name)
    try:model.export(target);actual=digest(target)
    finally:
        if target.exists():target.unlink()
    if actual!=digest(expected):raise ValueError(label+' checkpoint and native export weights differ')
    return actual


def _split_measurements(db,model,split):
    total={'policy_numerator':0.,'policy_weight':0.,'value_numerator':0.,'value_weight':0.,'rows':0,'value_rows':0}
    cursor=db.execute('SELECT id,payload FROM records WHERE split=? ORDER BY seq',(split,))
    identities=hashlib.sha256()
    while rows:=cursor.fetchmany(32):
        records=[json.loads(payload) for _,payload in rows]
        measured=measurements(model,records)
        if any(value is not None and isinstance(value,(int,float)) and not math.isfinite(value) for value in measured.values()):
            raise ValueError('nonfinite independent holdout metrics')
        for key in total:total[key]+=measured[key]
        for identity,_ in rows:identities.update((identity+'\n').encode())
    total['rows']=int(total['rows']);total['value_rows']=int(total['value_rows'])
    total['policy_loss']=total['policy_numerator']/total['policy_weight'] if total['policy_weight'] else None
    total['wdl_loss']=total['value_numerator']/total['value_weight'] if total['value_weight'] else None
    total['identity_sha256']=identities.hexdigest()
    return total


def _wdl(model,boards):
    values=[]
    with torch.no_grad():
        for start in range(0,len(boards),32):
            x=torch.from_numpy(np.stack([features(b) for b in boards[start:start+32]]))
            h=torch.relu(model.input(x))
            for layer in model.residual:h=torch.relu(h+layer(h))
            probabilities=torch.softmax(model.wdl(h),-1)
            if not bool(torch.isfinite(probabilities).all()):raise ValueError('nonfinite WDL predictions')
            values.extend(probabilities.tolist())
    return values


def start_consistency(model):
    """A one-ply value self-consistency diagnostic, not minimax correctness."""
    board=chess.Board();moves=sorted(board.legal_moves,key=lambda m:m.uci())
    children=[]
    for move in moves:
        child=board.copy(stack=True);child.push(move);children.append(child)
    root_wdl=_wdl(model,[board])[0];child_wdl=_wdl(model,children)
    with torch.no_grad():policy,_=model(torch.from_numpy(features(board)[None,:]))
    logits=policy[0,[action(m) for m in moves]];priors=torch.softmax(logits,-1).tolist()
    if not bool(torch.isfinite(logits).all()) or not all(math.isfinite(p) for p in priors):raise ValueError('nonfinite start policy')
    value=root_wdl[0]-root_wdl[2];rows=[]
    for move,wdl,prior in zip(moves,child_wdl,priors):
        backed_up=wdl[2]-wdl[0]
        rows.append({'uci':move.uci(),'san':board.san(move),'policy':prior,'child_wdl_side_to_move':wdl,
                     'root_value_from_child':backed_up,'absolute_parent_child_residual':abs(value-backed_up)})
    best=max(r['root_value_from_child'] for r in rows)
    return {'fen':board.fen(),'wdl':root_wdl,'value':value,'children':rows,'legal_moves':len(rows),
            'best_one_ply_backup_value':best,'absolute_bellman_residual':abs(value-best),
            'all_child_backups_negative':all(r['root_value_from_child']<0 for r in rows),
            'mean_absolute_parent_child_residual':sum(r['absolute_parent_child_residual'] for r in rows)/len(rows),
            'meaning':'side-to-move W-L with a sign flip at each ply; consistency is not labeled accuracy'}


def _actual_class(result,turn):
    value=RESULT_CLASS[result]
    return value if turn else 2-value


def outcome_metrics(model,rows):
    """Game-equal reliability against held-out actual human-game results."""
    if not rows:return {'rows':0,'games':0,'nll':None,'brier':None,'ece':None}
    counts=Counter(str(r['game_id']) for r in rows);bins=np.zeros((10,3),dtype=np.float64)
    sums={'nll':0.,'brier':0.};total=0.
    for start in range(0,len(rows),32):
        part=rows[start:start+32];boards=[chess.Board(r['fen']) for r in part]
        for row,board,probs in zip(part,boards,_wdl(model,boards)):
            label=_actual_class(row['result'],board.turn);weight=1./counts[str(row['game_id'])]
            sums['nll']-=weight*math.log(max(probs[label],1e-12))
            sums['brier']+=weight*sum((p-float(i==label))**2 for i,p in enumerate(probs))
            confidence=max(probs);choice=probs.index(confidence);bucket=min(9,int(confidence*10))
            bins[bucket]+=[weight,weight*confidence,weight*float(choice==label)];total+=weight
    return {'rows':len(rows),'games':len(counts),'nll':sums['nll']/total,'brier':sums['brier']/total,
            'ece':float(sum(abs(p-y) for n,p,y in bins if n)/total),
            'reliability_bins':[{'lower':i/10,'upper':(i+1)/10,'game_equivalent_weight':float(n),
                'mean_confidence':float(p/n) if n else None,'accuracy':float(y/n) if n else None}
                for i,(n,p,y) in enumerate(bins)],
            'weighting':'each completed game has total weight one',
            'domain':'actual held-out human-game outcomes; not perfect-play WDL or an Elo estimate'}


def _independent_outcomes(calibration,fixtures):
    path=Path(fixtures)/'official-calibration-rows.jsonl';meta_path=Path(str(path)+'.source.json')
    metadata=json.loads(meta_path.read_text(encoding='utf-8'))
    if metadata.get('license')!='CC0-1.0' or metadata.get('local_sha256')!=digest(path):
        raise ValueError('independent outcome source provenance/hash failed')
    groups=set(calibration.artifact['groups']['test']);rows=[]
    with path.open(encoding='utf-8') as source:
        for line in source:
            row=json.loads(line)
            if str(row['game_id']) in groups:
                if row['result'] not in RESULT_CLASS:raise ValueError('incomplete outcome in held-out calibration test')
                rows.append(row)
    return rows,metadata


def _leakage(db,calibration,puzzles):
    reserved=set(calibration.excluded_families)|set(puzzles.get('excluded_training_families',[]))
    if not reserved:raise ValueError('no independent evaluation families were reserved')
    tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    full='seen_families' in tables
    overlaps=[];families=0
    if full:
        for family,split in db.execute("SELECT family,split FROM seen_families WHERE split='train'"):
            families+=1
            if family in reserved:overlaps.append(family)
    retained=[]
    for family in db.execute("SELECT DISTINCT family FROM records WHERE split='train'"):
        if family[0] in reserved:retained.append(family[0])
    validation={r[0] for r in db.execute("SELECT DISTINCT family FROM records WHERE split='validation'")}
    test={r[0] for r in db.execute("SELECT DISTINCT family FROM records WHERE split='test'")}
    train={r[0] for r in db.execute("SELECT family FROM seen_families WHERE split='train'")} if full else {r[0] for r in db.execute("SELECT DISTINCT family FROM records WHERE split='train'")}
    split_overlap=sorted((train&validation)|(train&test)|(validation&test))
    return {'persistent_training_family_audit_available':full,'audited_training_families':families,
            'reserved_independent_families':len(reserved),'reserved_family_overlap':sorted(set(overlaps+retained)),
            'train_validation_test_family_overlap':split_overlap,
            'passed':full and not overlaps and not retained and not split_overlap,
            'scope':'new shard corpus only; inherited champion training history is unchanged'}


def _tactics(engine_path,model_path,puzzles,milliseconds=25,limit=20):
    selected=puzzles.get('test_positions',[])[:limit]
    if not selected:raise ValueError('official independent tactical positions are missing')
    results=[];correct=0;errors=0
    with Native(engine_path,model_path,timeout=20) as engine:
        if set(OPTIONS)-engine.options:raise ValueError('evaluation engine lacks required CHROMATIC search options')
        engine.pin_cpu(2);engine.configure(OPTIONS)
        if 'Threads' in engine.options:engine.configure({'Threads':1})
        for fixture in selected:
            board=chess.Board(fixture['fen_after_opponent_move']);expected=chess.Move.from_uci(fixture['expected_uci'])
            if not board.is_valid() or expected not in board.legal_moves:raise ValueError('invalid tactical fixture')
            result={'puzzle_id':fixture.get('puzzle_id'),'fen':board.fen(),'expected_uci':expected.uci(),'family':fixture['family']}
            try:
                answer=engine.play(board,milliseconds=milliseconds);move=chess.Move.from_uci(answer['bestmove'])
                legal=move in board.legal_moves
                if not legal:raise RuntimeError('illegal engine tactical move')
                hit=move==expected;correct+=hit
                result.update(answer,legal=True,exact_published_reply=hit)
            except Exception as exc:errors+=1;result.update(legal=False,error=type(exc).__name__+': '+str(exc),exact_published_reply=False)
            results.append(result)
    return {'positions':len(selected),'correct':correct,'accuracy':correct/len(selected),'engine_errors':errors,
            'milliseconds_per_position':milliseconds,'results':results,
            'metric':'exact published first reply; a different move can sometimes also be winning'}


def promotion_assessment(report):
    reasons=[];metrics=report['models'];candidate=metrics['candidate'];champion=metrics['champion']
    for split in ['validation','test']:
        a=candidate[split];b=champion[split]
        if a['rows']<1000:reasons.append(f'{split}_fewer_than_1000_fixed_positions')
        for key in ['policy_loss','wdl_loss']:
            if a[key] is None or b[key] is None or not math.isfinite(a[key]) or not math.isfinite(b[key]) or a[key]>b[key]+1e-9:
                reasons.append(f'{split}_{key}_missing_nonfinite_or_worse')
    if not report['calibration']['accepted']:reasons.append('calibrator_not_independently_accepted')
    for key in ['brier','ece']:
        a=candidate['outcome_calibration'][key];b=champion['outcome_calibration'][key]
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or a>b+1e-9:reasons.append('independent_outcome_'+key+'_missing_nonfinite_or_worse')
    if not report['source_integrity_verified']:reasons.append('original_training_frames_not_verified')
    if not report['leakage']['passed']:reasons.append('independent_evaluation_family_audit_failed_or_unavailable')
    if not report['candidate_head_stable']:reasons.append('candidate_head_changed_during_evaluation')
    for name in ['candidate','champion']:
        if metrics[name]['tactics']['engine_errors']:reasons.append(name+'_tactical_engine_errors')
    if candidate['tactics']['accuracy']<champion['tactics']['accuracy']:reasons.append('tactical_exact_reply_accuracy_worse')
    match=report['match']
    if match.get('completed_games',0)<64:reasons.append('fewer_than_64_completed_paired_games')
    if not match.get('complete') or match.get('engine_errors',0):reasons.append('match_incomplete_or_engine_error')
    if match.get('lower_bound') is None or match['lower_bound']<=.5:reasons.append('95_percent_score_lower_bound_not_above_half')
    return {'eligible':not reasons,'performed':False,'automatic_overwrite':False,'reasons':reasons,
            'meaning':'manual review only even if all independent gates pass; champion files are never written'}


def evaluate_run(data_directory,checkpoint_directory,engine_path,champion_pt,champion_zero,fixtures_directory,
                 pairs=2,milliseconds=10,max_plies=128):
    if not 1<=pairs<=128 or not 1<=milliseconds<=1000 or not 1<=max_plies<=512:raise ValueError('bounded evaluation budget exceeded')
    started=time.perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=Path(data_directory).resolve();checkpoint_dir=Path(checkpoint_directory).resolve();fixtures=Path(fixtures_directory).resolve()
    engine=Path(engine_path).resolve();champion_pt=Path(champion_pt).resolve();champion_zero=Path(champion_zero).resolve()
    before={'checkpoint':digest(champion_pt),'model':digest(champion_zero),'engine':digest(engine)}
    with closing(_connect(data)) as db:
        manifest_path=Path(_meta(db,'current_manifest') or '').resolve()
        if not manifest_path.is_relative_to(checkpoint_dir/'checkpoints'):raise ValueError('candidate manifest is outside this run checkpoint directory')
        manifest=read_manifest(manifest_path);state,mirror=verify_bundle(manifest_path)
        candidate_files={name:Path(spec['path']) for name,spec in manifest['files'].items()}
        candidate_before={name:digest(path) if path.is_file() else None for name,path in candidate_files.items()}
        manifest_before=digest(manifest_path)
        settings=_meta(db,'settings',state.get('settings',{}))
        calibration=Calibration.load(settings.get('calibration',fixtures/'calibration.json'))
        puzzle_path=fixtures/'official-puzzles.csv.source.json';puzzles=json.loads(puzzle_path.read_text(encoding='utf-8'))
        if puzzles.get('license')!='CC0-1.0':raise ValueError('tactical source is not official CC0')
        candidate_zero=Path(manifest['files']['model']['path']).resolve()
        candidate=_load_model(state);old_state,champion_actual=checkpoint.load(champion_pt);champion=_load_model(old_state)
        config={'manifest_sha256':digest(manifest_path),'candidate_zero_sha256':digest(candidate_zero),
                'champion':before,'pairs':pairs,'milliseconds':milliseconds,'max_plies':max_plies,'tactics_ms':25,'tactics_positions':20}
        evaluation_id=hashlib.sha256(canonical(config).encode()).hexdigest()[:24]
        output=checkpoint_dir/'evaluations'/evaluation_id;output.mkdir(parents=True,exist_ok=True)
        _export_matches(candidate,candidate_zero,output,'candidate');_export_matches(champion,champion_zero,output,'champion')
        outcome_rows,outcome_source=_independent_outcomes(calibration,fixtures)
        leak=_leakage(db,calibration,puzzles)
        model_reports={}
        for name,model in [('champion',champion),('candidate',candidate)]:
            model_reports[name]={split:_split_measurements(db,model,split) for split in ['validation','test']}
            model_reports[name]['start_position']=start_consistency(model)
            model_reports[name]['outcome_calibration']=outcome_metrics(model,outcome_rows)
        for split in ['validation','test']:
            if model_reports['candidate'][split]['identity_sha256']!=model_reports['champion'][split]['identity_sha256']:
                raise RuntimeError('fixed holdout changed during candidate evaluation')
        for name,path in [('champion',champion_zero),('candidate',candidate_zero)]:
            model_reports[name]['tactics']=_tactics(engine,path,puzzles)
        # Source code, time budget, affinity and search options are identical;
        # only the ModelFile differs. The arena preserves unfinished games as '*'.
        match=arena(str(engine),str(candidate_zero),str(champion_zero),output/'matches.json',pairs=pairs,
                    milliseconds=milliseconds,max_plies=max_plies,seed=2026,threshold=.55,confidence=.95,
                    candidate_options=OPTIONS,incumbent_options=OPTIONS,affinity_mask=2)
        atomic_json(checkpoint_dir/'matches.json',match)
        source=settings.get('source_provenance',{})
        source_verified=source.get('original_frame_integrity')=='verified_complete_zstd_frames'
        if str(settings.get('source','')).startswith('https://'):
            cursor=_meta(db,'source_cursor',{})
            source_verified=cursor.get('integrity')=='verified_complete_zstd_frames' and cursor.get('verified_frames',0)>0
        head_stable=_meta(db,'current_manifest')==str(manifest_path)
        candidate_after={name:digest(path) if path.is_file() else None for name,path in candidate_files.items()}
        if candidate_before!=candidate_after or digest(manifest_path)!=manifest_before:
            raise RuntimeError('candidate checkpoint or native weights changed during evaluation')
        report={'format':'chromatic-shard-evaluation-v1','evaluation_id':evaluation_id,'configuration':config,
                'candidate_manifest':str(manifest_path),'candidate_step':state['step'],'candidate_used_mirror':mirror,
                'candidate_zero':str(candidate_zero),'champion_checkpoint_actual':str(champion_actual),
                'candidate_head_stable':head_stable,'source_integrity_verified':source_verified,
                'candidate_preservation':{'before':candidate_before,'after':candidate_after,'unchanged':True,
                                          'manifest_sha256':manifest_before},
                'source_provenance':source,'leakage':leak,'models':model_reports,'match':match,
                'calibration':{'accepted':calibration.usable,'artifact_sha256':calibration.sha256,
                    'test_game_ids':sorted(calibration.artifact['groups']['test']),
                    'teacher_cp_mapping_test':calibration.artifact.get('reports',{}).get('test'),
                    'outcome_source':outcome_source,'meaning':'neural outcome ECE is separate from CP-mapping ECE and soft teacher-label loss'},
                'elapsed_seconds':time.perf_counter()-started}
    after={'checkpoint':digest(champion_pt),'model':digest(champion_zero),'engine':digest(engine)}
    if before!=after:raise RuntimeError('preserved champion or engine changed during evaluation')
    report['preservation']={'before':before,'after':after,'unchanged':True}
    report['promotion']=promotion_assessment(report)
    atomic_json(output/'evaluation-report.json',report);atomic_json(checkpoint_dir/'evaluation-report.json',report)
    return report


def common_holdout_report(common_data_directory,runs,champion_pt,output_path):
    """Compare every model on the very same fixed records and audit all train histories.

    runs maps display names to (data_directory, checkpoint_directory). No model,
    training ledger, checkpoint, or split assignment is modified.
    """
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=Path(output_path).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    champion_pt=Path(champion_pt).resolve();champion_hash=digest(champion_pt)
    old_state,actual=checkpoint.load(champion_pt)
    with closing(_connect(common_data_directory)) as common:
        common_families={split:{r[0] for r in common.execute('SELECT DISTINCT family FROM records WHERE split=?',(split,))}
                         for split in ['validation','test']}
        if not all(common_families.values()) or common_families['validation']&common_families['test']:
            raise ValueError('common holdout is empty or validation/test families overlap')
        results={'champion':{'trained_positions':0,'additional_training_steps':0,
                            'inherited_checkpoint_step':old_state['step'],'checkpoint':str(actual),
                            'checkpoint_sha256':champion_hash,
                            'training_family_audit':{'scope':'inherited champion history not available',
                                                     'new_training_positions':0},
                            **{split:_split_measurements(common,_load_model(old_state),split)
                               for split in ['validation','test']}}}
        for name,(data_directory,checkpoint_directory) in runs.items():
            if name=='champion':raise ValueError('candidate display name cannot be champion')
            checkpoint_directory=Path(checkpoint_directory).resolve()
            with closing(_connect(data_directory)) as db:
                head=Path(_meta(db,'current_manifest') or '').resolve()
                if not head.is_relative_to(checkpoint_directory/'checkpoints'):
                    raise ValueError('common comparison candidate manifest outside checkpoint directory')
                manifest=read_manifest(head);state,mirror=verify_bundle(head)
                inherited=state.get('settings',{}).get('initial_checkpoint')
                if inherited and digest(inherited)!=champion_hash:
                    raise ValueError('common comparison candidate did not inherit this champion')
                train={r[0] for r in db.execute("SELECT family FROM seen_families WHERE split='train'")}
                overlap={split:sorted(train&families) for split,families in common_families.items()}
                if any(overlap.values()):raise ValueError('common holdout leaked into '+name+' training history')
                model=_load_model(state)
                result={'trained_positions':state.get('progress',{}).get('positions_processed',0),
                        'additional_training_steps':state['step'],'candidate_manifest':str(head),
                        'candidate_used_mirror':mirror,'native_model_sha256':manifest['files']['model']['sha256'],
                        'training_family_audit':{'scope':'all persistent new training families',
                                                'families':len(train),'common_holdout_overlap':overlap,'passed':True},
                        **{split:_split_measurements(common,model,split) for split in ['validation','test']}}
                if _meta(db,'current_manifest')!=str(head):raise ValueError('candidate changed during common holdout evaluation')
                results[name]=result
        for split in ['validation','test']:
            hashes={r[split]['identity_sha256'] for r in results.values()}
            if len(hashes)!=1:raise RuntimeError('models were evaluated on different common holdout rows')
        report={'format':'chromatic-common-holdout-v1','data_directory':str(Path(common_data_directory).resolve()),
                'models':results,'common_families':{split:len(value) for split,value in common_families.items()},
                'meaning':'descriptive comparison on identical held-out records; training size and optimizer steps both vary',
                'inherited_champion_training_history':'not reconstructed; preserved unchanged'}
    if digest(champion_pt)!=champion_hash:raise RuntimeError('champion changed during common holdout evaluation')
    report['champion_unchanged']=True;atomic_json(output,report)
    columns=['model','trained_positions','additional_training_steps','split','rows','value_rows',
             'policy_loss','wdl_loss','policy_weight','value_weight','identity_sha256']
    csv_path=output.with_suffix('.csv');temporary=csv_path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as destination:
        writer=csv.DictWriter(destination,columns);writer.writeheader()
        for name,result in results.items():
            for split in ['validation','test']:
                writer.writerow({'model':name,'trained_positions':result['trained_positions'],
                    'additional_training_steps':result['additional_training_steps'],'split':split,
                    **{key:result[split][key] for key in columns[4:]}})
        destination.flush();os.fsync(destination.fileno())
    os.replace(temporary,csv_path)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='Evaluate a shard candidate without modifying the champion')
    parser.add_argument('--data-directory',required=True,type=Path);parser.add_argument('--checkpoint-directory',required=True,type=Path)
    parser.add_argument('--engine',required=True,type=Path);parser.add_argument('--champion-pt',required=True,type=Path)
    parser.add_argument('--champion-zero',required=True,type=Path);parser.add_argument('--fixtures-directory',required=True,type=Path)
    parser.add_argument('--pairs',type=int,default=2);parser.add_argument('--milliseconds',type=int,default=10);parser.add_argument('--max-plies',type=int,default=128)
    args=vars(parser.parse_args(argv));args['engine_path']=args.pop('engine')
    report=evaluate_run(**args)
    print(json.dumps({'evaluation_id':report['evaluation_id'],'candidate_step':report['candidate_step'],
          'promotion':report['promotion'],'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
