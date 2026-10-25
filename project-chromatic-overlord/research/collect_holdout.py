"""Bounded official CC0 capture; freeze a genuinely unused puzzle test split."""
import csv,hashlib,io,json,urllib.request
from pathlib import Path
import chess,zstandard
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
old=REPO/'chromatic-arc-trainer/fixtures/official-puzzles.csv'
oldrows=list(csv.DictReader(old.open(newline='')))
ids={r['PuzzleId'] for r in oldrows};games={r['GameUrl'].split('#')[0].rstrip('/').replace('/black','').replace('/white','') for r in oldrows}
def key(board):return ' '.join(board.fen(en_passant='legal').split()[:4])
seen=set()
for r in oldrows:
 b=chess.Board(r['FEN']);b.push_uci(r['Moves'].split()[0]);seen.add(key(b))
# Exclude exact positions from the available superset of training input too.
source=REPO/'chromatic-shard-trainer/fixtures/official-evaluations.jsonl.zst'
with source.open('rb') as f,zstandard.ZstdDecompressor().stream_reader(f) as reader:
 for line in io.TextIOWrapper(reader):
  r=json.loads(line);fen=r.get('fen',r.get('FEN'))
  if fen:
   try:seen.add(key(chess.Board(fen if len(fen.split())==6 else fen+' 0 1')))
   except ValueError:pass
for line in (REPO/'project-chromatic/runs/bootstrap/teacher.jsonl').read_text().splitlines():
 r=json.loads(line)
 if 'fen' in r:seen.add(key(chess.Board(r['fen'])))
url='https://database.lichess.org/lichess_db_puzzle.csv.zst';limit=1024*1024
request=urllib.request.Request(url,headers={'Range':f'bytes=0-{limit-1}','User-Agent':'Chromatic-factorial-research'})
response=urllib.request.urlopen(request,timeout=60)
class Bounded(io.RawIOBase):
 def __init__(self):self.count=0;self.h=hashlib.sha256()
 def readable(self):return True
 def readinto(self,buffer):
  n=min(len(buffer),limit-self.count)
  if n<=0:return 0
  data=response.read(n);buffer[:len(data)]=data;self.count+=len(data);self.h.update(data);return len(data)
raw=Bounded();selected=[];excluded=0
with zstandard.ZstdDecompressor().stream_reader(raw) as decoded:
 reader=csv.DictReader(io.TextIOWrapper(decoded,newline=''))
 for row in reader:
  game=row['GameUrl'].split('#')[0].rstrip('/').replace('/black','').replace('/white','')
  try:
   b=chess.Board(row['FEN']);moves=row['Moves'].split();b.push_uci(moves[0]);expected=moves[1]
   valid=b.is_valid() and chess.Move.from_uci(expected) in b.legal_moves
  except (ValueError,IndexError):valid=False
  if not valid or row['PuzzleId'] in ids or game in games or key(b) in seen:excluded+=1;continue
  selected.append(row);ids.add(row['PuzzleId']);games.add(game);seen.add(key(b))
  if len(selected)==200:break
response.close()
if len(selected)!=200:raise RuntimeError('Bounded capture did not yield 200 disjoint puzzles')
out=ROOT/'research/fixtures';out.mkdir(parents=True,exist_ok=True)
with (out/'holdout.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(selected[0]),lineterminator='\n');writer.writeheader();writer.writerows(selected)
meta={'source':url,'license':'CC0-1.0','license_evidence':'https://database.lichess.org/','status':response.status,'etag':response.headers.get('ETag'),'content_range':response.headers.get('Content-Range'),'compressed_bytes_read':raw.count,'compressed_prefix_sha256':raw.h.hexdigest(),'compressed_limit':limit,'decoding_started_at_byte':0,'frame_checksum':'not established: intentionally bounded prefix, verification was not disabled','positions':200,'excluded':excluded,'selection':'first 200 valid distinct-game and distinct-position puzzles outside old 100 and available training-input FENs, before any new experiment outcomes','old_fixture_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256((out/'holdout.csv').read_bytes()).hexdigest()}
(out/'holdout.source.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
