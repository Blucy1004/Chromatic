#!/usr/bin/env python3
"""Validate PGN/raw outcomes and reduce measured resource/strength evidence."""
import argparse,csv,gzip,json,statistics
from pathlib import Path
import chess,chess.pgn

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
 bench=json.loads((root/'search-bench.json').read_text());tactics=json.loads((root/'tactics.json').read_text())
 summary=json.loads((root/'summary.json').read_text());rows=[];validated=0
 for config,wdl in summary.items():
  path=root/f'{config}-games.json.gz'
  games=json.loads(gzip.decompress(path.read_bytes()) if path.exists() else (root/f'{config}-games.json').read_text())
  with (root/f'{config}.pgn').open() as f:
   parsed=[]
   while (game:=chess.pgn.read_game(f)) is not None:parsed.append(game)
  assert len(parsed)==len(games)==8
  for game,raw in zip(parsed,games):
   assert not game.errors
   assert [m.uci() for m in game.mainline_moves()]==raw['moves']
   assert game.headers['Result']==raw['result']
   assert raw['error'] is None,raw['error']
   outcome=game.end().board().outcome(claim_draw=True)
   assert (outcome.result() if outcome else '*')==raw['result']
   validated+=1
  b=[r for r in bench if r['config']==config];t=[r for r in tactics if r['config']==config]
  details=[r['detail'] for r in b]
  times=sorted(r['elapsed_ms'] for r in b)
  row={'config':config,**{k:wdl[k] for k in ['wins','draws','losses','unfinished','score']},
   'tactics_passed':sum(r['passed'] for r in t),'tactics_total':len(t),
   'median_search_nodes':statistics.median(r['simulations'] for r in details),
   'median_wall_ms':statistics.median(times),'max_wall_ms':max(times),
   'median_verification_nodes':statistics.median(r['verification_nodes'] for r in details),
   'fully_compared_searches':sum(r['verified_candidates']>=4 for r in details),
   'changed_search_choices':sum(r['verification_changed'] for r in details),
   'max_concurrent_leaves':max(r['peak_inflight'] for r in details),
   'median_parallel_batches':statistics.median(r['parallel_batches'] for r in details)}
  rows.append(row)
 with (root/'performance.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (root/'performance.json').write_text(json.dumps({'rows':rows,'validated_pgn_games':validated,'method':'12 timed position searches per configuration; 8 paired diagnostic tactics; not a rating benchmark'},indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
