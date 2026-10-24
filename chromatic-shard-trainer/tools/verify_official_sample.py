"""Replace the small eval fixture only after original zstd frames verify.

Hard caps: 60,000 rows, 16MiB compressed input, 128MiB decoded input,
128MiB single-frame spool, 256MiB total fixture-directory size.
No full-database download is performed.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys
import time
import zstandard

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.dont_write_bytecode=True
from shardtrain.streaming import RecordStream,SourceSpec,OFFICIAL_EVAL_URL,StreamLimit

MIB=1024*1024
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda:src.read(MIB),b''):h.update(chunk)
    return h.hexdigest()
def atomic_json(path,data):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as out:
        json.dump(data,out,indent=2);out.flush();os.fsync(out.fileno())
    os.replace(temp,path)

def main():
    started=time.perf_counter();directory=PROJECT/'fixtures';directory.mkdir(exist_ok=True)
    path=directory/'official-evaluations.jsonl.zst';sidecar=path.with_name(path.name+'.source.json')
    previous=json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
    temp=path.with_name(path.name+'.verified-new.tmp')
    def resource_check(extra):
        usage=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        if usage+extra>256*MIB:raise StreamLimit('small fixture total disk cap 256MiB reached')
        if time.perf_counter()-started>180:raise StreamLimit('small fixture elapsed cap 180 seconds reached')
    spec=SourceSpec(OFFICIAL_EVAL_URL,version=previous.get('etag','official bounded export'))
    expected={'location':OFFICIAL_EVAL_URL}
    if previous.get('etag'):expected['etag']=previous['etag']
    if previous.get('last_modified'):expected['last_modified']=previous['last_modified']
    count=0;selected=hashlib.sha256()
    try:
        with RecordStream(spec,spool_directory=directory/'verified-spool',max_compressed_bytes=16*MIB,
                          max_decompressed_bytes=128*MIB,max_frame_bytes=128*MIB,max_line_bytes=MIB,
                          expected_identity=expected,resource_check=resource_check,allow_partial_frames=False) as stream:
            lines=stream.iter_lines()
            try:
                with temp.open('wb') as raw:
                    with zstandard.ZstdCompressor(level=3,write_checksum=True).stream_writer(raw,closefd=False) as writer:
                        for number,text in lines:
                            row=json.loads(text)
                            if not isinstance(row,dict) or 'fen' not in row or 'evals' not in row:
                                raise RuntimeError(f'official schema mismatch at line {number}')
                            data=text.encode('utf-8')+b'\n';resource_check(len(data))
                            writer.write(data);selected.update(data);count+=1
                            if count==60000:break
                    raw.flush();os.fsync(raw.fileno())
            finally:lines.close()
            identity=dict(stream.identity);stats=dict(stream.stats);cursor=dict(stream.cursor)
        if count!=60000:raise RuntimeError(f'needed 60000 original verified-frame rows, received {count}')
        if stats['verified_frames']<1 or stats['partial_frames'] or stats['integrity']!='verified_complete_zstd_frames':
            raise RuntimeError('original full-frame verification was not established')
        same_rows=not previous.get('selected_jsonl_sha256') or previous['selected_jsonl_sha256']==selected.hexdigest()
        if not same_rows:raise RuntimeError('verified download differs from the previous selected row digest')
        local_sha=digest(temp)
        provenance={
            'url':OFFICIAL_EVAL_URL,'license':'CC0-1.0','license_evidence':'https://database.lichess.org/#evals',
            'version':identity.get('etag') or identity.get('last_modified'),'etag':identity.get('etag'),
            'last_modified':identity.get('last_modified'),'http_status':200,
            'content_length':identity.get('content_length'),'accept_ranges':identity.get('accept_ranges_advertised'),
            'compressed_start_offset':0,'compressed_bytes_read':stats['compressed_bytes'],
            'compressed_prefix_sha256':stats.get('compressed_prefix_sha256'),
            'decompressed_bytes_read':stats['decompressed_bytes'],'stop_reason':'verified_frame_record_budget',
            'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'intentionally_partial_original':True,
            'original_frame_checksum':'verified for every consumed standard frame; unseen full database not checked',
            'original_frame_integrity':'verified_complete_zstd_frames',
            'verified_original_frames':stats['verified_frames'],
            'verified_compressed_offset':stats['verified_compressed_offset'],
            'limits':{'compressed_bytes':16*MIB,'decompressed_bytes':128*MIB,'fixture_disk_bytes':256*MIB},
            'range_safety':'all decoding began at byte zero; live HTTP retries retain the decoder; no independent middle-offset decoding',
            'records':count,'source_line_start':1,'source_line_end':cursor['line'],
            'source_cursor':cursor,'selected_jsonl_sha256':selected.hexdigest(),
            'local_format':'JSONL/Zstandard','local_recompression_checksum_enabled':True,'local_sha256':local_sha}
        os.replace(temp,path);atomic_json(sidecar,provenance)
        manifest_path=directory/'official-sample-manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'));manifest['evaluations']=provenance
            atomic_json(manifest_path,manifest)
        result={'records':count,'source_identity':identity,'stream_stats':stats,'source_cursor':cursor,
                'same_selected_rows_as_previous_fixture':same_rows,
                'previous_local_sha256':previous.get('local_sha256'),'verified_local_sha256':local_sha,
                'elapsed_seconds':time.perf_counter()-started,'output':str(path)}
        atomic_json(PROJECT/'reports/official-frame-verification.json',result)
        print(json.dumps(result,indent=2))
    finally:
        if temp.exists():temp.unlink()

if __name__=='__main__':main()
