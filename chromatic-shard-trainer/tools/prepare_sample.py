"""Capture small, explicitly bounded official CC0 samples for integration tests.

This is a sample collector, not a whole-database downloader. Remote zstd streams
always start at compressed byte zero. Complete JSONL records / complete PGN games
are selected sequentially, then written as independent local files. A stopped
remote prefix does not establish its original frame checksum.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

import zstandard as zstd

PROJECT = Path(__file__).resolve().parents[1]
EVAL_URL = 'https://database.lichess.org/lichess_db_eval.jsonl.zst'
GAMES_URL = 'https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst'
PUZZLES_URL = 'https://database.lichess.org/lichess_db_puzzle.csv.zst'
OPENINGS_API = 'https://api.github.com/repos/lichess-org/chess-openings/commits/master'
MIB = 1024 * 1024
AGENT = 'PROJECT-CHROMATIC-small-CC0-validation/1.0'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.flush(); os.fsync(out.fileno())
    os.replace(temp, path)


def request(url, *, byte_limit, as_json=False):
    headers = {'User-Agent': AGENT, 'Accept-Encoding': 'identity'}
    if not as_json:
        headers['Range'] = f'bytes=0-{byte_limit-1}'
    else:
        headers['Accept'] = 'application/vnd.github+json'
    response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30)
    if response.status not in (200, 206):
        response.close(); raise RuntimeError(f'Unexpected HTTP status {response.status}')
    if response.status == 206:
        content_range = response.headers.get('Content-Range', '')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', content_range)
        if not match or int(match[1]) != 0:
            response.close(); raise RuntimeError('Range response did not start at compressed byte zero')
    return response


class CompressedBudget(io.RawIOBase):
    def __init__(self, response, limit):
        self.response=response; self.limit=limit; self.count=0; self.hash=hashlib.sha256()
        self.server_eof=False

    def readable(self): return True

    def read(self, size=-1):
        remaining=self.limit-self.count
        if remaining <= 0: return b''
        size=min(65536 if size < 0 else size, remaining)
        data=self.response.read(size)
        if not data:self.server_eof=True
        self.count+=len(data);self.hash.update(data)
        return data

    def readinto(self, target):
        data=self.read(len(target));target[:len(data)]=data;return len(data)


class RemoteLines:
    def __init__(self, url, compressed_limit, decompressed_limit):
        self.response=request(url,byte_limit=compressed_limit)
        self.raw=CompressedBudget(self.response,compressed_limit)
        # Installed 0.25.0 C backend enforces this argument in bytes for unknown-
        # size streaming frames (verified with a 4 MiB window regression).
        self.decoder=zstd.ZstdDecompressor(max_window_size=128*MIB).stream_reader(
            self.raw,read_size=65536,read_across_frames=True,closefd=False)
        self.limit=decompressed_limit;self.decompressed=0;self.reason='consumer_limit'

    def lines(self, max_line_bytes=512*1024):
        pending=bytearray()
        while self.decompressed < self.limit:
            try:
                block=self.decoder.read(min(65536,self.limit-self.decompressed))
            except zstd.ZstdError:
                if self.raw.count >= self.raw.limit:
                    self.reason='compressed_budget';return
                raise
            if not block:
                self.reason='compressed_budget' if self.raw.count>=self.raw.limit else 'remote_end'
                return
            self.decompressed+=len(block);pending.extend(block)
            while True:
                end=pending.find(b'\n')
                if end<0:break
                if end+1>max_line_bytes:raise RuntimeError('Source line exceeded bounded line length')
                line=bytes(pending[:end+1]);del pending[:end+1]
                yield line
            if len(pending)>max_line_bytes:raise RuntimeError('Source line exceeded bounded line length')
        self.reason='decompressed_budget'

    def provenance(self, url):
        headers=self.response.headers
        version=headers.get('ETag') or headers.get('Last-Modified') or 'unversioned-prefix'
        return {'url':url,'license':'CC0-1.0','license_evidence':'https://database.lichess.org/',
                'version':version,'etag':headers.get('ETag'),'last_modified':headers.get('Last-Modified'),
                'http_status':self.response.status,'content_range':headers.get('Content-Range'),
                'content_length':headers.get('Content-Length'),'accept_ranges':headers.get('Accept-Ranges'),
                'compressed_start_offset':0,'compressed_bytes_read':self.raw.count,
                'compressed_prefix_sha256':self.raw.hash.hexdigest(),
                'decompressed_bytes_read':self.decompressed,'stop_reason':self.reason,
                'captured_at_utc':datetime.now(timezone.utc).isoformat(),
                'intentionally_partial_original':True,
                'original_frame_checksum':'not established for intentionally stopped remote prefix',
                'limits':{'compressed_bytes':self.raw.limit,'decompressed_bytes':self.limit},
                'range_safety':'all decoding began at compressed byte zero; no arbitrary offset decoding'}

    def close(self):
        self.decoder.close();self.response.close()


def reusable(path):
    meta=path.with_name(path.name+'.source.json')
    if path.exists() and meta.exists():
        data=json.loads(meta.read_text(encoding='utf-8'))
        if digest(path)==data['local_sha256']:
            print(json.dumps({'reuse':str(path),'records':data.get('records'),'games':data.get('games')}),flush=True)
            return data
    return None


def capture_evals(directory, positions=60000, compressed_limit=16*MIB):
    path=directory/'official-evaluations.jsonl.zst'
    previous=reusable(path)
    if previous:return previous
    remote=RemoteLines(EVAL_URL,compressed_limit,128*MIB)
    temp=path.with_name(path.name+'.tmp');count=0;line_hash=hashlib.sha256()
    try:
        with temp.open('wb') as target:
            with zstd.ZstdCompressor(level=3,write_checksum=True).stream_writer(target,closefd=False) as writer:
                for line in remote.lines():
                    value=json.loads(line)
                    if not isinstance(value,dict) or 'fen' not in value or 'evals' not in value:
                        raise RuntimeError('Official evaluation JSONL schema mismatch')
                    writer.write(line);line_hash.update(line);count+=1
                    if count>=positions:
                        remote.reason='record_budget';break
            target.flush();os.fsync(target.fileno())
        if count<12000:raise RuntimeError(f'Insufficient bounded evaluation sample: {count} raw records')
        provenance=remote.provenance(EVAL_URL)
        provenance.update(records=count,source_line_start=1,source_line_end=count,
                          selected_jsonl_sha256=line_hash.hexdigest(),local_format='JSONL/Zstandard',
                          local_recompression_checksum_enabled=True)
        os.replace(temp,path);provenance['local_sha256']=digest(path)
        atomic_json(path.with_name(path.name+'.source.json'),provenance)
        print(json.dumps({'eval_records':count,'compressed_bytes_read':provenance['compressed_bytes_read'],
                          'decompressed_bytes_read':provenance['decompressed_bytes_read'],'output':str(path)}),flush=True)
        return provenance
    finally:remote.close()


def capture_games(directory, games=5000, compressed_limit=8*MIB):
    path=directory/'official-games-sample.pgn'
    previous=reusable(path)
    if previous:return previous
    remote=RemoteLines(GAMES_URL,compressed_limit,128*MIB)
    temp=path.with_name(path.name+'.tmp');count=0;pending=bytearray();raw_lines=0
    complete_end=re.compile(rb'(?:1-0|0-1|1/2-1/2|\*)\s*$')
    try:
        with temp.open('wb') as target:
            for line in remote.lines():
                raw_lines+=1;pending.extend(line)
                if len(pending)>2*MIB:raise RuntimeError('PGN game exceeded 2 MiB safety bound')
                if not line.lstrip().startswith(b'[') and complete_end.search(line.rstrip()):
                    target.write(pending);target.write(b'\n');pending.clear();count+=1
                    if count>=games:
                        remote.reason='complete_game_budget';break
            target.flush();os.fsync(target.fileno())
        if count<100:raise RuntimeError(f'Insufficient bounded complete PGN games: {count}')
        provenance=remote.provenance(GAMES_URL)
        provenance.update(games=count,source_lines_read=raw_lines,source_line_start=1,
                          complete_game_only=True,discarded_incomplete_game_bytes=len(pending),
                          local_format='PGN UTF-8')
        os.replace(temp,path);provenance['local_sha256']=digest(path)
        atomic_json(path.with_name(path.name+'.source.json'),provenance)
        print(json.dumps({'games':count,'compressed_bytes_read':provenance['compressed_bytes_read'],
                          'decompressed_bytes_read':provenance['decompressed_bytes_read'],'output':str(path)}),flush=True)
        return provenance
    finally:remote.close()


def tiny_fetch(url,limit,*,as_json=False):
    with request(url,byte_limit=limit,as_json=as_json) as response:
        body=response.read(limit+1)
        if len(body)>limit:raise RuntimeError(f'Bounded text file too large: {url}')
        if response.status==206:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if not match or int(match[3])!=len(body):raise RuntimeError(f'Text source was truncated by sample budget: {url}')
        return body,dict(response.headers)


def capture_openings(directory):
    manifest_path=directory/'official-openings.source.json'
    if manifest_path.exists():
        meta=json.loads(manifest_path.read_text(encoding='utf-8'))
        if all((directory/name).is_file() and digest(directory/name)==row['sha256'] for name,row in meta['files'].items()):
            print(json.dumps({'reuse':'official-openings','commit':meta['version']}),flush=True);return meta
    commit_body,_=tiny_fetch(OPENINGS_API,256*1024,as_json=True)
    commit=json.loads(commit_body)['sha']
    if not re.fullmatch('[0-9a-f]{40}',commit):raise RuntimeError('Invalid official openings commit')
    files={};total=0
    for name in ['COPYING.txt','a.tsv','b.tsv','c.tsv','d.tsv','e.tsv']:
        url=f'https://raw.githubusercontent.com/lichess-org/chess-openings/{commit}/{name}'
        body,headers=tiny_fetch(url,2*MIB-total)
        total+=len(body)
        if total>2*MIB:raise RuntimeError('Openings corpus exceeded total small-sample limit')
        if name=='COPYING.txt' and b'CC0' not in body and b'Creative Commons' not in body:
            raise RuntimeError('Official openings CC0 license text missing')
        path=directory/('official-openings-'+name);path.write_bytes(body)
        files[path.name]={'url':url,'sha256':digest(path),'bytes':len(body),'etag':headers.get('ETag')}
    meta={'license':'CC0-1.0','url':'https://github.com/lichess-org/chess-openings','version':commit,
          'license_evidence':f'https://raw.githubusercontent.com/lichess-org/chess-openings/{commit}/COPYING.txt',
          'files':files,'total_bytes':total,'captured_at_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(manifest_path,meta)
    print(json.dumps({'openings_bytes':total,'commit':commit}),flush=True);return meta


def extract_calibration(directory,provenance):
    sys.path.insert(0,str(PROJECT))
    from shardtrain.calibration import iter_pgn_calibration_rows
    path=directory/'official-calibration-rows.jsonl'
    previous=reusable(path)
    if previous:
        if previous.get('source_file_sha256')!=provenance['local_sha256']:
            raise RuntimeError('Existing calibration snapshot belongs to another PGN source; choose a fresh output directory')
        return previous
    temp=path.with_name(path.name+'.tmp');count=0;groups=set();families=set()
    with (directory/'official-games-sample.pgn').open(encoding='utf-8') as source,temp.open('w',encoding='utf-8',newline='\n') as target:
        for row in iter_pgn_calibration_rows(source,provenance,max_games=5000,max_rows=20000,min_ply=13,max_rows_per_game=8):
            target.write(json.dumps(row,separators=(',',':'))+'\n');count+=1;groups.add(row['game_id']);families.add(row['family'])
        target.flush();os.fsync(target.fileno())
    os.replace(temp,path)
    meta={'license':'CC0-1.0','url':provenance['url'],'version':provenance['version'],
          'source_file_sha256':provenance['local_sha256'],'local_sha256':digest(path),'records':count,
          'games':len(groups),'families':len(families),'source':'official game outcomes plus existing PGN cp annotations',
          'independent_from_eval_training':True,'minimum_ply':13,'maximum_rows_per_game':8}
    atomic_json(path.with_name(path.name+'.source.json'),meta)
    print(json.dumps({'calibration_rows':count,'calibration_games':len(groups),'output':str(path)}),flush=True)
    return meta


def capture_puzzles(directory, rows=100):
    import chess
    path=directory/'official-puzzles.csv'
    previous=reusable(path)
    if previous:return previous
    remote=RemoteLines(PUZZLES_URL,MIB,4*MIB)
    temp=path.with_name(path.name+'.tmp');count=0;families=set();positions=[]
    try:
        records=csv.DictReader(line.decode('utf-8') for line in remote.lines())
        required={'PuzzleId','FEN','Moves'}
        if not required<=set(records.fieldnames or []):raise RuntimeError('Official puzzle CSV schema mismatch')
        with temp.open('w',encoding='utf-8',newline='') as target:
            writer=csv.DictWriter(target,fieldnames=records.fieldnames);writer.writeheader()
            for row in records:
                board=chess.Board(row['FEN']);moves=row['Moves'].split()
                if not board.is_valid() or len(moves)<2:raise RuntimeError('Invalid puzzle position/solution')
                first=board.parse_uci(moves[0]);board.push(first)
                expected=board.parse_uci(moves[1])
                if expected not in board.legal_moves:raise RuntimeError('Puzzle first answer is illegal')
                family=hashlib.sha256(board.board_fen().encode()).hexdigest();families.add(family)
                positions.append({'puzzle_id':row['PuzzleId'],'fen_after_opponent_move':board.fen(en_passant='legal'),
                                  'expected_uci':expected.uci(),'family':family})
                writer.writerow(row);count+=1
                if count>=rows:remote.reason='puzzle_record_budget';break
            target.flush();os.fsync(target.fileno())
        if count!=rows:raise RuntimeError(f'Insufficient bounded puzzle sample: {count}')
        provenance=remote.provenance(PUZZLES_URL)
        provenance.update(records=count,local_format='CSV UTF-8',source_record_start=1,source_record_end=count,
                          role='independent tactical test only; excluded from neural training',
                          first_opponent_and_expected_reply_legality_checked=True,
                          expected_match_metric='exact published solution first reply; alternatives may also win',
                          excluded_training_families=sorted(families),test_positions=positions)
        os.replace(temp,path);provenance['local_sha256']=digest(path)
        atomic_json(path.with_name(path.name+'.source.json'),provenance)
        print(json.dumps({'puzzles':count,'compressed_bytes_read':provenance['compressed_bytes_read'],'output':str(path)}),flush=True)
        return provenance
    finally:remote.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'fixtures')
    parser.add_argument('--eval-records',type=int,default=60000)
    parser.add_argument('--games',type=int,default=5000)
    parser.add_argument('--puzzles-only',action='store_true',help='Capture only 100 official tactical-test rows; leave eval/game snapshots untouched')
    args=parser.parse_args()
    if not 12000<=args.eval_records<=60000 or not 100<=args.games<=5000:
        raise SystemExit('Small sample guard: 12,000..60,000 eval rows and 100..5,000 games only')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.puzzles_only:
        capture_puzzles(args.output,100)
        print('OFFICIAL_PUZZLE_SAMPLE_READY',flush=True)
        return
    eval_meta=capture_evals(args.output,args.eval_records)
    game_meta=capture_games(args.output,args.games)
    openings_meta=capture_openings(args.output)
    calibration_meta=extract_calibration(args.output,game_meta)
    atomic_json(args.output/'official-sample-manifest.json',{'evaluations':eval_meta,'games':game_meta,
                'openings':openings_meta,'calibration':calibration_meta,
                'download_guards':{'eval_compressed_bytes':16*MIB,'games_compressed_bytes':8*MIB,
                                   'decompressed_bytes_each':128*MIB,'openings_bytes_total':2*MIB}})
    print('OFFICIAL_SMALL_SAMPLE_READY',flush=True)


if __name__=='__main__':main()
