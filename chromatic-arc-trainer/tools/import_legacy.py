"""Read-only legacy-run snapshot import into a new, empty ARC run directory.

The original database is opened immutable+read-only. A nonempty WAL is rejected
instead of checkpointing the source. Only the destination is ever modified.
Trusted local CHROMATIC checkpoints contain pickle; do not import unknown files.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from shardtrain.storage import atomic_json,canonical,digest
from shardtrain.durable import FORMAT,OWNED,read_manifest,verify_bundle,_required_shards_exist

IMPORT_FORMAT='chromatic-legacy-import-v1'
EXECUTION={'device':'cpu','resolved_device':'cpu','cpu_threads':1,'precision':'fp32','prefetch':0,
           'cache_mode':'auto','transfer_mode':'copy','gradient_accumulation':1,'optimizer':'default','fallback':'error'}

def _meta(db,key,default=None):
    row=db.execute('SELECT v FROM meta WHERE k=?',(key,)).fetchone()
    return json.loads(row[0]) if row else default

def _put(db,key,value):db.execute('INSERT OR REPLACE INTO meta(k,v) VALUES (?,?)',(key,canonical(value)))

def _owned(path,root,pattern=None):
    original=Path(path)
    if not original.is_absolute() or original.is_symlink():raise ValueError('Source paths must be absolute owned regular files')
    resolved=original.resolve()
    if resolved.parent!=root or (pattern and not pattern.fullmatch(resolved.name)):
        raise ValueError('Source path escaped its owned directory: '+str(path))
    return resolved

def _wal_empty(database):
    wal=Path(str(database)+'-wal')
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError('Source has a nonempty WAL. Stop the original trainer and restore a clean closed snapshot; this tool never checkpoints or writes the source.')

def _fixed_file(entry,root):
    path=_owned(entry['path'],root,re.compile(r'^(validation|test)\.jsonl$'))
    if not path.is_file() or digest(path)!=entry['sha256']:raise ValueError('Frozen legacy holdout is missing or corrupt')
    identities=[]
    with path.open(encoding='utf-8') as stream:
        for line in iter(lambda:stream.readline(2*1024**2+1),''):
            if len(line)>2*1024**2:raise ValueError('Legacy normalized line exceeds safety limit')
            identities.append(json.loads(line)['identity'])
    if identities!=entry['ids']:raise ValueError('Frozen legacy holdout order differs from ledger')
    return path

def import_legacy(source_data_directory,source_checkpoint_directory,destination_root,*,max_copy_gb=4.,dry_run=False):
    """Copy a consistent legacy trainer-v1 run, retaining all training state.

    Returned target data/candidate are ready for the ARC owner's narrowly
    checked settings migration. PT bytes and their old settings/signature are
    intentionally unchanged. Source data/calibration/initial paths stay put.
    """
    source_data=Path(source_data_directory).resolve();source_checkpoints=Path(source_checkpoint_directory).resolve()
    destination=Path(destination_root).resolve();target_data=destination/'data';target_checkpoints=destination/'candidate'
    if not math.isfinite(float(max_copy_gb)) or max_copy_gb<=0:raise ValueError('MaxCopyGB must be finite and positive')
    for source in [source_data,source_checkpoints]:
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError('Destination and original directories must not contain one another')
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError('Destination must be new or empty; partial imports are never reused')
    database=source_data/'ledger.sqlite3'
    if not database.is_file() or not source_checkpoints.is_dir():raise ValueError('Legacy source data/checkpoint directories do not exist')
    _wal_empty(database)
    source_db_sha=digest(database)
    source_files={database:source_db_sha}
    bundle_root=source_checkpoints/'checkpoints'
    if bundle_root.resolve()!=bundle_root:raise ValueError('Legacy checkpoint directory is a symlink')
    uri=database.as_uri()+'?mode=ro&immutable=1'
    src=sqlite3.connect(uri,uri=True)
    dst=None;started=False
    try:
        src.execute('BEGIN')
        if src.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Legacy SQLite integrity check failed')
        settings=_meta(src,'settings');legacy_signature=_meta(src,'settings_signature')
        if not isinstance(settings,dict) or settings.get('trainer_version')!=1:
            raise ValueError('Only a legacy shard trainer version-1 ledger is supported')
        if hashlib.sha256(canonical(settings).encode()).hexdigest()!=legacy_signature:raise ValueError('Legacy settings signature does not match')
        if Path(settings['data_directory']).resolve()!=source_data or Path(settings['checkpoint_directory']).resolve()!=source_checkpoints:
            raise ValueError('Legacy settings do not belong to the selected source directories')
        owner=source_checkpoints/'checkpoint-owner.json'
        if not owner.is_file() or json.loads(owner.read_text(encoding='utf-8'))!={'format':FORMAT,'ledger_token':_meta(src,'durable_owner_token')}:
            raise ValueError('Legacy checkpoint owner does not match the selected ledger')
        source_files[owner]=digest(owner)
        frozen=_meta(src,'fixed_sets')
        if not isinstance(frozen,dict) or set(frozen)!= {'validation','test'}:raise ValueError('Legacy fixed holdouts are not committed')
        for entry in frozen.values():
            path=_fixed_file(entry,(source_data/'fixed').resolve());source_files[path]=digest(path)
        current_manifest=_owned(_meta(src,'current_manifest'),bundle_root,OWNED)
        original_head_seq=_meta(src,'current_checkpoint_seq')
        rows=src.execute('SELECT seq,manifest FROM checkpoints ORDER BY seq DESC').fetchall()
        if not rows or not isinstance(original_head_seq,int):raise ValueError('No committed legacy checkpoint head')
        if not any(seq==original_head_seq and Path(path).resolve()==current_manifest for seq,path in rows):
            raise ValueError('Legacy checkpoint head disagrees with committed rows')
        manifests={};path_mapping={};copy_pairs=[];errors=[];selected=None
        def map_manifest(path):
            owned=_owned(path,bundle_root,OWNED)
            if not owned.name.endswith('.manifest.json'):raise ValueError('Owned receipt is not a manifest')
            result=str(target_checkpoints/'checkpoints'/owned.name);path_mapping[str(owned)]=result
            return result
        for seq,path in rows:
            manifest_path=_owned(path,bundle_root,OWNED)
            if not manifest_path.name.endswith('.manifest.json'):raise ValueError('Checkpoint row is not a bundle manifest')
            map_manifest(manifest_path)
            if not manifest_path.is_file():continue
            source_files[manifest_path]=digest(manifest_path)
            manifest=read_manifest(manifest_path);manifests[seq]=(manifest_path,manifest)
            for name in ('checkpoint','mirror','model'):
                file=_owned(manifest['files'][name]['path'],bundle_root,OWNED)
                if file.is_file():
                    source_files[file]=digest(file);copy_pairs.append((file,target_checkpoints/'checkpoints'/file.name))
            if seq>original_head_seq:continue
            if selected is None:
                try:
                    state,mirror=verify_bundle(manifest_path)
                    if state.get('settings_signature')!=legacy_signature or state.get('settings')!=settings:
                        raise ValueError('Legacy checkpoint and ledger settings differ')
                    _required_shards_exist(SimpleNamespace(db=src),state)
                    selected=(seq,manifest_path,state,mirror)
                except Exception as exc:errors.append(f'{seq}: {type(exc).__name__}: {exc}')
        if selected is None:raise RuntimeError('No verified recoverable legacy checkpoint: '+'; '.join(errors))
        selected_seq,selected_manifest,state,used_mirror=selected
        shard_rows=src.execute('SELECT id,path,sha,rows,status,checkpoint FROM shards ORDER BY id').fetchall()
        shard_mapping={}
        for shard_id,path,expected,count,status,receipt in shard_rows:
            file=_owned(path,(source_data/'shards').resolve(),re.compile(r'^shard-\d{8}\.jsonl$'))
            if file.name!=f'shard-{shard_id:08d}.jsonl':raise ValueError('Legacy shard ID and owned filename differ')
            shard_mapping[shard_id]=str(target_data/'shards'/file.name)
            if file.is_file():
                actual=digest(file)
                if actual!=expected:raise ValueError('Retained legacy shard checksum mismatch')
                source_files[file]=actual;copy_pairs.append((file,target_data/'shards'/file.name))
            if receipt is not None:map_manifest(receipt)
        # Immutable external inputs are deliberately not relocated: the stream
        # cursor's source identity must continue to refer to exactly that file.
        external_inputs={}
        for key,sha_key in [('initial_checkpoint','initial_sha256'),('calibration','calibration_sha256')]:
            if key in settings:
                file=Path(settings[key])
                if not file.is_file() or (settings.get(sha_key) and digest(file)!=settings[sha_key]):
                    raise ValueError('Legacy immutable input differs: '+key)
                external_inputs[file]=digest(file)
        source=Path(settings['source']) if not str(settings['source']).startswith('https://') else None
        if source is not None:
            if not source.is_file():raise ValueError('Legacy source is missing from its original path')
            expected=settings.get('source_provenance',{}).get('local_sha256')
            if expected and digest(source)!=expected:raise ValueError('Legacy source hash differs from provenance')
            external_inputs[source]=digest(source)
        expected_bytes=sum(path.stat().st_size for path in source_files)
        if expected_bytes>int(max_copy_gb*1024**3):raise RuntimeError('Import exceeds MaxCopyGB; source unchanged')
        summary={'format':IMPORT_FORMAT,'source_data_directory':str(source_data),'source_checkpoint_directory':str(source_checkpoints),
            'target_data_directory':str(target_data),'target_checkpoint_directory':str(target_checkpoints),
            'original_head_seq':original_head_seq,'imported_head_seq':selected_seq,'imported_step':state['step'],
            'source_head_used_mirror':used_mirror,'source_head_rollback':selected_seq!=original_head_seq,'source_verification_errors':errors,
            'source_database_sha256':source_db_sha,'source_head_manifest_sha256':digest(selected_manifest),
            'source_files_verified':len(source_files),'copy_bytes_upper_estimate':expected_bytes,
            'training_started':False,'source_modified':False,'dry_run':bool(dry_run)}
        if dry_run:
            _wal_empty(database)
            if any(not file.is_file() or digest(file)!=expected for file,expected in {**source_files,**external_inputs}.items()):
                raise RuntimeError('Source changed during read-only import preflight')
            return summary
        destination.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(destination.parent).free<expected_bytes+16*1024**2:raise RuntimeError('Insufficient destination free space')
        destination.mkdir(exist_ok=True)
        if any(destination.iterdir()):raise ValueError('Destination changed during preflight; refusing import')
        started=True;atomic_json(destination/'import-incomplete.json',{'format':IMPORT_FORMAT,'phase':'copying','source_modified':False})
        for directory in [target_data/'fixed',target_data/'shards',target_checkpoints/'checkpoints']:directory.mkdir(parents=True,exist_ok=True)
        dst=sqlite3.connect(target_data/'ledger.sqlite3',isolation_level=None)
        src.backup(dst)
        dst.execute('PRAGMA journal_mode=DELETE')
        for entry in frozen.values():
            file=Path(entry['path']);copy_pairs.append((file,target_data/'fixed'/file.name))
        copied=[]
        for file,target in dict(copy_pairs).items():
            expected=source_files[file]
            shutil.copy2(file,target)
            if digest(target)!=expected:raise RuntimeError('Imported file checksum differs: '+str(target))
            copied.append({'source':str(file),'target':str(target),'sha256':expected,'bytes':target.stat().st_size})
        for seq,(manifest_path,manifest) in manifests.items():
            mapped=copy.deepcopy(manifest)
            for spec in mapped['files'].values():spec['path']=str(target_checkpoints/'checkpoints'/Path(spec['path']).name)
            atomic_json(Path(path_mapping[str(manifest_path)]),mapped)
        new_head=path_mapping[str(selected_manifest)]
        mapped_frozen=copy.deepcopy(frozen)
        for entry in mapped_frozen.values():entry['path']=str(target_data/'fixed'/Path(entry['path']).name)
        runtime=copy.deepcopy(_meta(src,'runtime_config',settings))
        runtime.update(EXECUTION,trainer_version=2,data_directory=str(target_data),checkpoint_directory=str(target_checkpoints),
            benchmark_output=str(ROOT/'reports/benchmarks'),resume=True,dry_run=False,benchmark=False,profile=False)
        token=uuid.uuid4().hex
        marker={**summary,'status':'pending_arc_migration','copy_complete':True,'legacy_settings':settings,
            'legacy_settings_signature':legacy_signature,'required_execution':dict(EXECUTION),
            'imported_head_manifest':new_head,'legacy_checkpoint_bytes_preserved':True,
            'original_source_paths_preserved':True,'destination_owner_token':token}
        dst.execute('BEGIN IMMEDIATE')
        try:
            for seq,path in rows:
                if seq<=selected_seq and seq in manifests:dst.execute('UPDATE checkpoints SET manifest=? WHERE seq=?',(path_mapping[str(Path(path).resolve())],seq))
                else:dst.execute('DELETE FROM checkpoints WHERE seq=?',(seq,))
            completed=set(state['progress']['completed'])
            for shard_id,path,expected,count,status,receipt in shard_rows:
                mapped_receipt=path_mapping[str(Path(receipt).resolve())] if receipt else None
                if selected_seq!=original_head_seq:
                    status='done' if shard_id in completed else 'prepared';mapped_receipt=new_head if shard_id in completed else None
                dst.execute('UPDATE shards SET path=?,checkpoint=?,status=? WHERE id=?',(shard_mapping[shard_id],mapped_receipt,status,shard_id))
            for event_id,payload in dst.execute('SELECT id,payload FROM events').fetchall():
                event=json.loads(payload)
                if event.get('commit_seq',0)>selected_seq:dst.execute('DELETE FROM events WHERE id=?',(event_id,));continue
                for key in ['checkpoint','latest_checkpoint']:
                    if event.get(key):
                        resolved=str(Path(event[key]).resolve())
                        event[key]=map_manifest(resolved)
                dst.execute('UPDATE events SET payload=? WHERE id=?',(canonical(event),event_id))
            _put(dst,'fixed_sets',mapped_frozen);_put(dst,'runtime_config',runtime);_put(dst,'durable_owner_token',token)
            _put(dst,'current_manifest',new_head);_put(dst,'current_checkpoint_seq',selected_seq);_put(dst,'current_global_step',state['step'])
            _put(dst,'recovered_seq',selected_seq if selected_seq!=original_head_seq or used_mirror else None)
            dst.execute('COMMIT')
        except BaseException:dst.execute('ROLLBACK');raise
        atomic_json(target_checkpoints/'checkpoint-owner.json',{'format':FORMAT,'ledger_token':token})
        if dst.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise RuntimeError('Copied SQLite integrity failed')
        copied_state,copied_mirror=verify_bundle(new_head)
        if copied_state['step']!=state['step'] or copied_state['settings_signature']!=legacy_signature:
            raise RuntimeError('Copied checkpoint state changed')
        for entry in mapped_frozen.values():_fixed_file(entry,target_data/'fixed')
        _wal_empty(database)
        for file,expected in {**source_files,**external_inputs}.items():
            if not file.is_file() or digest(file)!=expected:raise RuntimeError('Source changed while import was copying; partial destination retained and source never written')
        _wal_empty(database)
        final_bytes=sum(file.stat().st_size for file in destination.rglob('*') if file.is_file())
        if final_bytes>int(max_copy_gb*1024**3):raise RuntimeError('Imported files exceed MaxCopyGB; partial destination retained')
        # The migration authorization appears only after copied files, SQLite,
        # frozen data, checkpoint state, and unchanged source all verified.
        dst.execute('BEGIN IMMEDIATE')
        try:_put(dst,'legacy_import',marker);dst.execute('COMMIT')
        except BaseException:dst.execute('ROLLBACK');raise
        summary.update(status='imported-pending-arc-migration',imported_head_manifest=new_head,
            legacy_settings_signature=legacy_signature,copied_files=copied,verified_source_unchanged=True,
            pt_optimizer_scheduler_rng_replay_progress='byte-identical checkpoint copies',destination_owner_token=token,imported_bytes=final_bytes)
        atomic_json(destination/'import-report.json',summary)
        (destination/'import-incomplete.json').unlink()
        dst.close();dst=None
        return summary
    except BaseException as exc:
        if started:
            if dst is not None:
                try:_put(dst,'legacy_import_failed',{'format':IMPORT_FORMAT,'error':f'{type(exc).__name__}: {exc}'})
                except Exception:pass
            atomic_json(destination/'import-incomplete.json',{'format':IMPORT_FORMAT,'phase':'failed','error':f'{type(exc).__name__}: {exc}','source_modified':False})
        raise
    finally:
        if dst is not None:dst.close()
        src.close()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data-directory',type=Path,required=True)
    parser.add_argument('--source-checkpoint-directory',type=Path,required=True)
    parser.add_argument('--destination-root',type=Path,required=True)
    parser.add_argument('--max-copy-gb',type=float,default=4.)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    result=import_legacy(args.source_data_directory,args.source_checkpoint_directory,args.destination_root,
                         max_copy_gb=args.max_copy_gb,dry_run=args.dry_run)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
