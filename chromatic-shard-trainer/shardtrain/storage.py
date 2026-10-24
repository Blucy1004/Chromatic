from __future__ import annotations
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:f.write(canonical(obj)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

class ResourceGuard:
    def __init__(self,roots,max_disk_gb,max_memory_gb):
        paths=sorted({Path(p).resolve() for p in roots},key=lambda p:len(p.parts))
        self.roots=[]
        for p in paths:
            if not any(p.is_relative_to(q) for q in self.roots):self.roots.append(p)
        self.disk_limit=int(max_disk_gb*1024**3);self.memory_limit=int(max_memory_gb*1024**3)
        self.last_disk=0;self.last_scan=0;self.peak_rss=0;self.pending_reservation=0
    def disk(self):
        total=0
        for root in self.roots:
            if root.exists():
                for p in root.rglob('*'):
                    if p.is_file() and not p.is_symlink():
                        try:total+=p.stat().st_size
                        except FileNotFoundError:pass
        self.last_disk=total;self.last_scan=time.monotonic();self.pending_reservation=0;return total
    def check(self,extra_disk=0,force=False):
        import psutil
        rss=psutil.Process().memory_info().rss;self.peak_rss=max(self.peak_rss,rss)
        if rss>self.memory_limit:raise RuntimeError(f'MaxMemoryGB exceeded: RSS {rss} > {self.memory_limit}')
        disk=self.disk() if force or time.monotonic()-self.last_scan>1 or self.pending_reservation+extra_disk>8*1024**2 else self.last_disk
        required=disk+self.pending_reservation+extra_disk+2*1024**2
        if required>self.disk_limit:
            disk=self.disk();required=disk+extra_disk+2*1024**2
        if required>self.disk_limit:raise RuntimeError(f'MaxDiskGB exceeded: reserved {required} > {self.disk_limit}; durable data retained')
        self.pending_reservation+=extra_disk
        return rss,disk

class Ledger:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=15,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA cache_size=-8192');self.db.execute('PRAGMA busy_timeout=15000')
        self.db.execute('PRAGMA wal_autocheckpoint=256');self.db.execute('PRAGMA journal_size_limit=2097152')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS seen(id TEXT PRIMARY KEY,bucket TEXT,phase TEXT,split TEXT);
        CREATE INDEX IF NOT EXISTS seen_bucket ON seen(bucket);
        CREATE TABLE IF NOT EXISTS seen_families(family TEXT PRIMARY KEY,split TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records(seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE,family TEXT,bucket TEXT,phase TEXT,split TEXT,payload TEXT,shard INTEGER);
        CREATE INDEX IF NOT EXISTS pool ON records(split,phase,shard,seq);
        CREATE TABLE IF NOT EXISTS shards(id INTEGER PRIMARY KEY,path TEXT,sha TEXT,rows INTEGER,status TEXT,checkpoint TEXT);
        CREATE TABLE IF NOT EXISTS checkpoints(seq INTEGER PRIMARY KEY AUTOINCREMENT,manifest TEXT UNIQUE,step INTEGER,completed INTEGER,active INTEGER);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,payload TEXT NOT NULL);
        ''')
    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:yield;self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise
    def get(self,key,default=None):
        row=self.db.execute('SELECT v FROM meta WHERE k=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default
    def put(self,key,value):self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,canonical(value)))
    def rows(self,split=None,shard=None,phase=None,unassigned=False):
        sql='SELECT id,payload FROM records WHERE 1=1';args=[]
        for key,val in [('split',split),('shard',shard),('phase',phase)]:
            if val is not None:sql+=' AND '+key+'=?';args.append(val)
        if unassigned:sql+=' AND shard IS NULL'
        sql+=' ORDER BY seq'
        for identity,payload in self.db.execute(sql,args):yield identity,json.loads(payload)
    def load_ids(self,ids):
        result=[]
        for identity in ids:
            row=self.db.execute('SELECT payload FROM records WHERE id=?',(identity,)).fetchone()
            if row is None:raise RuntimeError('Missing active shard row '+identity+'; recovery refuses to skip data')
            result.append(json.loads(row[0]))
        return result
    def count(self,split,phase=None,unassigned=False):
        sql='SELECT count(*) FROM records WHERE split=?';args=[split]
        if phase is not None:sql+=' AND phase=?';args.append(phase)
        if unassigned:sql+=' AND shard IS NULL'
        return self.db.execute(sql,args).fetchone()[0]
    def event(self,obj):self.db.execute('INSERT INTO events(payload) VALUES (?)',(canonical(obj),))
    def export_logs(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        def entries():
            for i,p in self.db.execute('SELECT id,payload FROM events ORDER BY id'):yield dict(json.loads(p),event_id=i)
        for suffix in ['jsonl','csv']:
            path=directory/('training.'+suffix);temp=path.with_suffix(path.suffix+'.tmp')
            with temp.open('w',encoding='utf-8',newline='') as f:
                if suffix=='jsonl':
                    for obj in entries():f.write(canonical(obj)+'\n')
                else:
                    fields=list(dict.fromkeys(k for obj in entries() for k in obj))
                    if fields:
                        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(entries())
                f.flush();os.fsync(f.fileno())
            os.replace(temp,path)
    def close(self):
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.db.close()

def test_hook(name,checkpoint_dir,step,shard):
    if os.environ.get('CHROMATIC_TEST_PAUSE_AT')!=name:return
    if step<int(os.environ.get('CHROMATIC_TEST_PAUSE_STEP','1')):return
    if shard<int(os.environ.get('CHROMATIC_TEST_PAUSE_SHARD','0')):return
    atomic_json(Path(checkpoint_dir)/'test-marker.json',{'hook':name,'global_step':step,'shard':shard,'pid':os.getpid()})
    while True:time.sleep(.1)
