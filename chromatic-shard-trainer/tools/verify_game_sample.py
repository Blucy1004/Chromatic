"""Verify the original PGN frame prefix without changing the captured games.

Hard limits: 8MiB compressed, 64MiB decoded, 64MiB frame spool,
256MiB total fixture directory, 180 seconds, 5,000 complete PGN records.
"""
from pathlib import Path
import datetime
import hashlib
import json
import re
import sys
import time
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.dont_write_bytecode=True
from shardtrain.streaming import RecordStream,SourceSpec,StreamLimit
from shardtrain.storage import atomic_json,digest

MIB=1024*1024
URL='https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst'
END=re.compile(r'(?:1-0|0-1|1/2-1/2|\*)\s*$')
SITE=re.compile(r'^\[Site "https://lichess\.org/([A-Za-z0-9]{8})(?:/[^\"]*)?"\]$')
EVAL=re.compile(r'\[%eval ([^\]]*)\]')


def labels(lines):
    sites=hashlib.sha256();evaluations=hashlib.sha256();site_count=0;eval_count=0
    for line in lines:
        match=SITE.fullmatch(line.rstrip('\r\n'))
        if match:sites.update((match[1]+'\n').encode());site_count+=1
        for value in EVAL.findall(line):evaluations.update((value+'\n').encode());eval_count+=1
    return {'game_ids':site_count,'game_ids_sha256':sites.hexdigest(),
            'eval_annotations':eval_count,'eval_annotations_sha256':evaluations.hexdigest()}


def main():
    started=time.perf_counter();directory=PROJECT/'fixtures';path=directory/'official-games-sample.pgn'
    sidecar=Path(str(path)+'.source.json');previous=json.loads(sidecar.read_text(encoding='utf-8'))
    original_hash=digest(path)
    if previous['local_sha256']!=original_hash:raise ValueError('captured PGN changed from its provenance')
    with path.open(encoding='utf-8') as existing:expected_labels=labels(existing)
    preserved={str(p):digest(p) for p in [path,directory/'official-calibration-rows.jsonl',directory/'calibration.json']}
    expected={'location':URL,'etag':previous['etag'],'last_modified':previous['last_modified']}
    selected=hashlib.sha256();sites=hashlib.sha256();evaluations=hashlib.sha256()
    games=0;site_count=0;eval_count=0;pending_bytes=0;identity={};stats={};cursor={};error=None
    def resource_check(extra):
        usage=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        if usage+extra>256*MIB:raise StreamLimit('small fixture total disk cap 256MiB reached')
        if time.perf_counter()-started>180:raise StreamLimit('small fixture elapsed cap 180 seconds reached')
    stream=None
    try:
        source=SourceSpec(URL,original_url=URL,version=previous['version'])
        with RecordStream(source,spool_directory=directory/'verified-spool',max_compressed_bytes=8*MIB,
                          max_decompressed_bytes=64*MIB,max_frame_bytes=64*MIB,max_line_bytes=2*MIB,
                          max_window_kib=64*1024,expected_identity=expected,resource_check=resource_check,
                          allow_partial_frames=False) as stream:
            lines=stream.iter_lines()
            try:
                for number,text in lines:
                    raw=text.encode('utf-8')+b'\n';selected.update(raw);pending_bytes+=len(raw)
                    if pending_bytes>2*MIB:raise StreamLimit('PGN game exceeded 2MiB capture bound')
                    site=SITE.fullmatch(text)
                    if site:sites.update((site[1]+'\n').encode());site_count+=1
                    for value in EVAL.findall(text):evaluations.update((value+'\n').encode());eval_count+=1
                    if not text.lstrip().startswith('[') and END.search(text.rstrip()):
                        # Match the original capture's extra record separator exactly.
                        selected.update(b'\n');pending_bytes=0;games+=1
                        if games==5000:break
            finally:lines.close()
            identity=dict(stream.identity);stats=dict(stream.stats);cursor=dict(stream.cursor)
        actual_labels={'game_ids':site_count,'game_ids_sha256':sites.hexdigest(),
                       'eval_annotations':eval_count,'eval_annotations_sha256':evaluations.hexdigest()}
        if games!=5000 or selected.hexdigest()!=original_hash or actual_labels!=expected_labels:
            raise RuntimeError('verified original PGN differs from captured games or evaluation annotations')
        if stats['partial_frames'] or stats['verified_frames']<1 or stats['verified_frames_without_checksum']:
            raise RuntimeError('complete original frames with checksums were not established')
        evidence={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'source_identity':identity,'stream_stats':stats,'source_cursor':cursor,
                  'selected_pgn_sha256':selected.hexdigest(),'identical_game_ids_and_eval_annotations':True,
                  'labels':actual_labels,'complete_games':games,'complete_original_frames_consumed':stats['verified_frames'],
                  'scope':'every consumed original standard frame checked by decoder checksum; unseen rest of database not downloaded or verified'}
        updated={**previous,'original_frame_checksum':'verified for every consumed original standard frame; unseen rest of database not checked',
                 'original_frame_integrity':'verified_complete_zstd_frames','verified_original_frames':stats['verified_frames'],
                 'verified_compressed_offset':stats['verified_compressed_offset'],'frame_verification':evidence}
        # The source PGN and all derived calibration artifacts remain byte-identical.
        if any(digest(p)!=sha for p,sha in preserved.items()):raise RuntimeError('preserved PGN or calibration artifact changed')
        atomic_json(sidecar,updated)
        manifest_path=directory/'official-sample-manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'));manifest['games']=updated;atomic_json(manifest_path,manifest)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        if stream is not None:identity=dict(stream.identity);stats=dict(stream.stats);cursor=dict(stream.cursor)
    report={'passed':error is None,'error':error,'source_identity':identity,'stream_stats':stats,'source_cursor':cursor,
            'games':games,'same_local_sha256':selected.hexdigest()==original_hash,'local_sha256':original_hash,
            'labels':expected_labels,'preserved_artifacts':preserved,
            'limits':{'compressed_bytes':8*MIB,'decompressed_bytes':64*MIB,'frame_bytes':64*MIB,
                      'fixture_disk_bytes':256*MIB,'elapsed_seconds':180},
            'elapsed_seconds':time.perf_counter()-started,'source_sidecar_updated':error is None}
    atomic_json(PROJECT/'reports/game-frame-verification.json',report)
    print(json.dumps(report,indent=2));return 0 if error is None else 1

if __name__=='__main__':raise SystemExit(main())
