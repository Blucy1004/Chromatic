"""Transactional bounded queues, frozen holdouts and rotating shard files."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

import chess

from .quality import (QualityConfig, QualityError, identity_for, family_for,
                      similarity_bucket_for, board_from_fen, parse_record)
from .storage import canonical, digest, test_hook
from .streaming import RecordStream, SourceSpec

PHASES=('opening','middlegame','endgame')
MIB=1024*1024


class InsufficientData(RuntimeError):
    pass


def quotas(size):
    size=int(size)
    return {phase:size//3+int(i<size%3) for i,phase in enumerate(PHASES)}


class ShardManager:
    def __init__(self,ledger,data_dir,guard,config,calibration,source_spec,anchors,excluded_families=None):
        self.ledger=ledger;self.data_dir=Path(data_dir).resolve();self.guard=guard;self.config=dict(config)
        self.calibration=calibration;self.source_spec=source_spec if isinstance(source_spec,SourceSpec) else SourceSpec(str(source_spec))
        self.shard_dir=self.data_dir/'shards';self.fixed_dir=self.data_dir/'fixed';self.spool_dir=self.data_dir/'source-spool'
        for path in (self.shard_dir,self.fixed_dir,self.spool_dir):
            path.mkdir(parents=True,exist_ok=True)
            if path.resolve()!=path:raise RuntimeError('Managed shard directories cannot be symlinks')
        self.seed=int(self.config.get('seed',2026));self.shard_size=int(self.config.get('shard_size',10000))
        self.similarity_cap=int(self.config.get('similarity_cap',50))
        if self.shard_size<1 or self.similarity_cap<1:raise ValueError('ShardSize and similarity cap must be positive')
        positions=int(self.config.get('positions',100000))
        self.fixed_sizes={'validation':int(self.config.get('validation_size',max(16,min(1000,positions//10)))),
                          'test':int(self.config.get('test_size',max(16,min(1000,positions//10))))}
        if min(self.fixed_sizes.values())<3:raise ValueError('Fixed validation/test sets need at least one row per phase')
        self.quality_config=QualityConfig(min_depth=int(self.config.get('min_depth',12)),
                                          min_knodes=int(self.config.get('min_knodes',10)))
        self.excluded_families=set(excluded_families or ())
        if calibration is not None:self.excluded_families.update(calibration.excluded_families)
        self.root_bucket=similarity_bucket_for(chess.Board())
        self.stream=None;self.iterator=None;self.broken=False
        self._cleanup_owned_temps()
        self._initialize_anchors(list(anchors))

    def _cleanup_owned_temps(self):
        # The pipeline holds its exclusive data-directory lock before creating
        # a manager. Only these private, uncommitted scratch names are eligible.
        rules=((self.spool_dir,re.compile(r'^\.source-frame-[A-Za-z0-9_-]+\.jsonl\.tmp$')),
               (self.shard_dir,re.compile(r'^shard-\d{8}\.jsonl\.tmp$')),
               (self.fixed_dir,re.compile(r'^(validation|test)\.jsonl\.tmp$')))
        removed=0
        for directory,pattern in rules:
            for path in directory.iterdir():
                if path.is_symlink() or not path.is_file() or not pattern.fullmatch(path.name):continue
                resolved=path.resolve()
                if resolved.parent!=directory or not resolved.is_relative_to(self.data_dir):
                    raise RuntimeError('Temporary-file cleanup escaped the managed data directory')
                path.unlink();removed+=1
        if removed:
            with self.ledger.transaction():
                counts=Counter(self.ledger.get('quality_stats',{}));counts['orphan_temps_removed']+=removed
                self.ledger.put('quality_stats',dict(counts))

    def _split(self,bucket):
        if bucket==self.root_bucket:return 'train'
        value=int(hashlib.sha256(f'{self.seed}:{bucket}'.encode()).hexdigest(),16)%100
        return 'train' if value<80 else 'validation' if value<90 else 'test'

    def _capacity(self,split,phase):
        if split=='train':return 2*self.shard_size
        return quotas(self.fixed_sizes[split])[phase]

    def _insert(self,row,counts):
        db=self.ledger.db;identity=row['identity'];family=row['family'];bucket=row['similarity_bucket'];phase=row['phase']
        if phase not in PHASES:raise ValueError('unknown normalized phase')
        split=self._split(bucket)
        if db.execute('SELECT 1 FROM seen WHERE id=?',(identity,)).fetchone():counts['duplicates']+=1;return False
        # Mark every examined valid identity, including rejected/overflow rows,
        # so later copies cannot change split membership or enter a future queue.
        bucket_count=db.execute('SELECT count(*) FROM seen WHERE bucket=?',(bucket,)).fetchone()[0]
        db.execute('INSERT INTO seen(id,bucket,phase,split) VALUES (?,?,?,?)',(identity,bucket,phase,split))
        if family in self.excluded_families:counts['excluded_calibration_or_tactical_family']+=1;return False
        if bucket_count>=self.similarity_cap:counts['similarity_capped']+=1;return False
        if self.ledger.count(split,phase,unassigned=True)>=self._capacity(split,phase):
            counts['queue_overflow_'+split]+=1;return False
        previous_split=db.execute('SELECT split FROM seen_families WHERE family=?',(family,)).fetchone()
        if previous_split and previous_split[0]!=split:
            raise RuntimeError('Position family crossed a fixed data split')
        db.execute('INSERT OR IGNORE INTO seen_families(family,split) VALUES (?,?)',(family,split))
        payload=canonical(dict(row,split=split))
        # SQLite pages, indexes and WAL need more than the raw JSON payload.
        self.guard.check(extra_disk=len(payload.encode('utf-8'))*3+2048)
        db.execute('INSERT INTO records(id,family,bucket,phase,split,payload,shard) VALUES (?,?,?,?,?,?,NULL)',
                   (identity,family,bucket,phase,split,payload))
        counts['accepted_'+split]+=1;counts['accepted_'+split+'_'+phase]+=1
        return True

    def _initialize_anchors(self,anchors):
        identities=[r['identity'] for r in anchors]
        signature=hashlib.sha256(canonical(identities).encode()).hexdigest()
        previous=self.ledger.get('anchors_initialized')
        if previous:
            if previous.get('sha256')!=signature:raise RuntimeError('Opening anchor catalog/seed changed during resume')
            return
        root_identity=identity_for(chess.Board())
        root=next((r for r in anchors if r.get('identity')==root_identity),None)
        if root is None:raise RuntimeError('Opening anchors must include the start position')
        if root['family'] in self.excluded_families:
            raise RuntimeError('Start-position anchor conflicts with independent calibration/test exclusion')
        ordered=[root]+[r for r in anchors if r['identity']!=root_identity]
        counts=Counter(self.ledger.get('quality_stats',{}))
        with self.ledger.transaction():
            for row in ordered:
                if row.get('provenance',{}).get('license')!='CC0-1.0':raise ValueError('Opening anchor lacks official CC0 provenance')
                if self._insert(row,counts):counts['anchors_inserted']+=1
            found=self.ledger.db.execute('SELECT split FROM records WHERE id=?',(root_identity,)).fetchone()
            if not found or found[0]!='train':raise RuntimeError('The starting position was not retained for training')
            self.ledger.put('anchors_initialized',{'sha256':signature,'rows':len(anchors),'root_identity':root_identity})
            self.ledger.put('quality_stats',dict(counts))

    def _start(self):
        if self.broken:raise RuntimeError('Source ingestion failed; close and resume from the committed cursor')
        if self.iterator is not None:return
        cursor=self.ledger.get('source_cursor',{})
        identity=self.ledger.get('source_identity') or cursor.get('source_identity')
        self.stream=RecordStream(self.source_spec,spool_directory=self.spool_dir,
            max_compressed_bytes=int(self.config.get('max_download_mb',128))*MIB,
            max_decompressed_bytes=int(self.config.get('max_decompressed_mb',256))*MIB,
            max_frame_bytes=int(self.config.get('max_frame_mb',128))*MIB,
            expected_identity=identity,resource_check=lambda extra:self.guard.check(extra_disk=extra),
            allow_partial_frames=False)
        self.iterator=iter(self.stream.iter_lines(start_line=int(cursor.get('line',0)),
                                                 expected_prefix_sha256=cursor.get('prefix_sha256')))

    def _ingest_chunk(self):
        if self.ledger.get('source_exhausted',False):return False
        self._start();counts=Counter(self.ledger.get('quality_stats',{}));raw_count=0;exhausted=False
        try:
            with self.ledger.transaction():
                for _ in range(128):
                    self.guard.check()
                    try:line,text=next(self.iterator)
                    except StopIteration:exhausted=True;break
                    raw_count+=1;counts['raw_lines']+=1
                    try:
                        raw=json.loads(text)
                        if not isinstance(raw,dict):raise ValueError('JSONL value is not an object')
                    except (ValueError,TypeError):counts['invalid_json']+=1;continue
                    source={'license':self.source_spec.license,'url':self.source_spec.original_url,
                            'version':self.source_spec.version,'line':line,
                            'source_identity_sha256':hashlib.sha256(canonical(self.stream.identity).encode()).hexdigest()}
                    try:row=parse_record(raw,self.calibration,self.quality_config,source=source)
                    except QualityError as exc:counts['rejected_'+exc.reason]+=1;continue
                    self._insert(row,counts)
                # Cursor and every accepted/rejected line decision commit in the
                # same SQLite transaction. An interrupted chunk replays safely.
                self.ledger.put('source_cursor',dict(self.stream.cursor))
                self.ledger.put('source_identity',dict(self.stream.identity))
                self.ledger.put('source_stats',dict(self.stream.stats))
                self.ledger.put('quality_stats',dict(counts))
                if exhausted:self.ledger.put('source_exhausted',True)
            return raw_count>0
        except BaseException:
            self.broken=True
            if self.stream:self.stream.close()
            raise

    def _enough(self,split,size):
        return all(self.ledger.count(split,phase,unassigned=True)>=n for phase,n in quotas(size).items())

    def _fill(self,requirements):
        while not all(self._enough(split,size) for split,size in requirements.items()):
            if not self._ingest_chunk():
                shortages={split:{phase:max(0,n-self.ledger.count(split,phase,unassigned=True))
                                  for phase,n in quotas(size).items()} for split,size in requirements.items()}
                raise InsufficientData('Official source ended before balanced quotas could be filled: '+canonical(shortages))

    def _select_ids(self,split,size):
        chosen=[]
        for phase,n in quotas(size).items():
            chosen.extend(self.ledger.db.execute('SELECT seq,id FROM records WHERE split=? AND phase=? AND shard IS NULL ORDER BY seq LIMIT ?',
                                                   (split,phase,n)).fetchall())
        if len(chosen)!=size:raise InsufficientData('Balanced queue selection was short')
        return [identity for _,identity in sorted(chosen)]

    def _write_rows(self,path,ids):
        path=Path(path);temp=path.with_name(path.name+'.tmp')
        estimated=sum(self.ledger.db.execute('SELECT length(CAST(payload AS BLOB)) FROM records WHERE id=?',(identity,)).fetchone()[0]+1 for identity in ids)
        self.guard.check(extra_disk=estimated)
        with temp.open('w',encoding='utf-8',newline='\n') as stream:
            for identity in ids:
                row=self.ledger.db.execute('SELECT payload FROM records WHERE id=?',(identity,)).fetchone()
                if row is None:raise RuntimeError('Normalized record disappeared while writing a shard')
                stream.write(row[0]+'\n')
            stream.flush();os.fsync(stream.fileno())
        if path.exists() and digest(path)!=digest(temp):
            raise RuntimeError('Refusing to overwrite a normalized file with different contents: '+str(path))
        os.replace(temp,path)
        return digest(path)

    def _fixed_rows(self,entry):
        path=Path(entry['path'])
        if not path.is_file() or digest(path)!=entry['sha256']:
            raise RuntimeError('Frozen holdout file is missing or corrupted: '+str(path))
        rows=self.ledger.load_ids(entry['ids'])
        if len(rows)!=len(entry['ids']):raise RuntimeError('Frozen holdout membership changed')
        return rows

    def prepare_fixed(self):
        frozen=self.ledger.get('fixed_sets')
        if frozen:
            return self._fixed_rows(frozen['validation']),self._fixed_rows(frozen['test'])
        self._fill(self.fixed_sizes)
        frozen={}
        for split,size in self.fixed_sizes.items():
            ids=self._select_ids(split,size);path=self.fixed_dir/(split+'.jsonl')
            frozen[split]={'ids':ids,'path':str(path),'sha256':self._write_rows(path,ids)}
        with self.ledger.transaction():self.ledger.put('fixed_sets',frozen)
        return self._fixed_rows(frozen['validation']),self._fixed_rows(frozen['test'])

    def _safe_shard_path(self,path,shard_id):
        path=Path(path).resolve()
        if not path.is_relative_to(self.shard_dir.resolve()):
            raise RuntimeError('Refusing shard path outside the managed shard directory')
        if path.parent!=self.shard_dir or path.name!=f'shard-{int(shard_id):08d}.jsonl':
            raise RuntimeError('Refusing a path that does not match its owned shard ID')
        return path

    def _prepared(self,row):
        shard_id,path,sha,expected=row;path=self._safe_shard_path(path,shard_id)
        stored=self.ledger.db.execute('SELECT id FROM records WHERE shard=? ORDER BY seq',(shard_id,)).fetchall()
        if not path.exists():
            if len(stored)!=expected:raise RuntimeError('Prepared shard and normalized recovery rows are both missing')
            actual=self._write_rows(path,[r[0] for r in stored])
            if actual!=sha:raise RuntimeError('Reconstructed prepared shard differs from its committed hash')
        if digest(path)!=sha:raise RuntimeError('Prepared shard checksum mismatch; it is retained for inspection')
        if len(stored)!=expected:
            # Rebuild the entire assigned order, never append missing rows out of
            # order and silently alter an active epoch permutation.
            restored_ids=[]
            with self.ledger.transaction(),path.open(encoding='utf-8') as stream:
                self.ledger.db.execute('DELETE FROM records WHERE shard=?',(shard_id,))
                for line in iter(lambda:stream.readline(2*MIB+1),''):
                    if len(line)>2*MIB:raise RuntimeError('Normalized shard line exceeded safety bound')
                    record=json.loads(line);board=board_from_fen(record['fen'])
                    if record['identity']!=identity_for(board):raise RuntimeError('Normalized shard identity mismatch')
                    self.ledger.db.execute('INSERT INTO records(id,family,bucket,phase,split,payload,shard) VALUES (?,?,?,?,?,?,?)',
                        (record['identity'],record['family'],record['similarity_bucket'],record['phase'],'train',canonical(record),shard_id))
                    restored_ids.append(record['identity'])
                    self.guard.check()
                if len(restored_ids)!=expected:raise RuntimeError('Prepared shard row count mismatch')
            stored=[(identity,) for identity in restored_ids]
        return {'id':shard_id,'path':str(path),'sha256':sha,'ids':[r[0] for r in stored],'rows':expected}

    def next_shard(self,remaining):
        remaining=int(remaining)
        if remaining<=0:return None
        row=self.ledger.db.execute("SELECT id,path,sha,rows FROM shards WHERE status IN ('prepared','ready','training') ORDER BY id LIMIT 1").fetchone()
        if row:
            if row[3]>remaining:raise RuntimeError('Prepared shard exceeds remaining Positions; resume configuration changed')
            return self._prepared(row)
        size=min(remaining,self.shard_size);self._fill({'train':size})
        ids=self._select_ids('train',size)
        shard_id=self.ledger.db.execute('SELECT COALESCE(max(id),0)+1 FROM shards').fetchone()[0]
        path=self.shard_dir/f'shard-{shard_id:08d}.jsonl';sha=self._write_rows(path,ids)
        with self.ledger.transaction():
            self.ledger.db.execute('INSERT INTO shards(id,path,sha,rows,status,checkpoint) VALUES (?,?,?,?,?,NULL)',
                                  (shard_id,str(path),sha,len(ids),'prepared'))
            self.ledger.db.executemany('UPDATE records SET shard=? WHERE id=?',((shard_id,identity) for identity in ids))
        return {'id':shard_id,'path':str(path),'sha256':sha,'ids':ids,'rows':len(ids)}

    def cleanup(self,completed,keep=False):
        completed={int(i) for i in completed};db=self.ledger.db
        # Verified checkpoint progress is authoritative after a corruption
        # fallback; ahead-of-checkpoint success receipts must not skip training.
        with self.ledger.transaction():
            for shard_id,status in db.execute('SELECT id,status FROM shards').fetchall():
                if status=='done' and shard_id not in completed:
                    db.execute("UPDATE shards SET status='prepared',checkpoint=NULL WHERE id=?",(shard_id,))
        if keep or not completed:return
        last=max(completed)
        receipt=db.execute('SELECT status,checkpoint FROM shards WHERE id=?',(last,)).fetchone()
        if not receipt or receipt[0]!='done' or not receipt[1]:
            raise RuntimeError('Completion receipt is not committed; retaining all shard recovery data')
        eligible=db.execute("SELECT id,path,checkpoint,sha FROM shards WHERE status='done' AND id<? ORDER BY id",(last,)).fetchall()
        for shard_id,path,checkpoint,expected_sha in eligible:
            if shard_id not in completed or not checkpoint:continue
            target=self._safe_shard_path(path,shard_id)
            test_hook('before_delete',self.config.get('checkpoint_directory',self.data_dir),
                      int(self.ledger.get('current_global_step',0)),int(shard_id))
            if target.exists():
                if digest(target)!=expected_sha:raise RuntimeError('Completed shard changed; retaining it for inspection')
                target.unlink()
            with self.ledger.transaction():
                # Replay payloads are embedded in the committed checkpoint;
                # fixed validation/test rows and global dedup keys are retained.
                db.execute("DELETE FROM records WHERE split='train' AND shard=?",(shard_id,))
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        db.execute('PRAGMA incremental_vacuum(64)')
        self.guard.check(force=True)

    def close(self):
        # Closing the iterator unwinds the verified-frame spool generator and
        # releases its open file before Windows directory cleanup/restart.
        if self.iterator is not None and hasattr(self.iterator,'close'):
            self.iterator.close()
        if self.stream:self.stream.close()
        self.iterator=None;self.stream=None

