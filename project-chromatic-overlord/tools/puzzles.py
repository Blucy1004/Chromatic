#!/usr/bin/env python3
"""Independent published Lichess first-reply diagnostics; no model training."""
import argparse,csv,json,math,os
from pathlib import Path
import chess
from experiment import Engine,CONFIGS,ROOT,MODEL,sha,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Choose a new result file; previous puzzles remain preserved')
 if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
 fixture=ROOT.parent/'chromatic-arc-trainer/fixtures/official-puzzles.csv'
 fixtures=[]
 with fixture.open(newline='') as stream:
  for row in csv.DictReader(stream):
   b=chess.Board(row['FEN']);moves=row['Moves'].split();b.push_uci(moves[0]);expected=moves[1]
   assert b.is_valid() and chess.Move.from_uci(expected) in b.legal_moves
   fixtures.append({'id':row['PuzzleId'],'fen':b.fen(en_passant='fen'),'expected':expected,'rating':int(row['Rating'])})
 results=[];summary=[]
 for ms,names in [(30,list(CONFIGS)),(100,['disabled','tactical','combination'])]:
  for name in names:
   with Engine(a.engine,CONFIGS[name]) as e:
    for row in fixtures:
     r=e.search(row['fen'],ms=ms);legal=chess.Move.from_uci(r['move']) in chess.Board(row['fen']).legal_moves
     assert legal
     results.append({'config':name,'ms':ms,**row,'legal':legal,'hit':r['move']==row['expected'],**r})
   selected=[r for r in results if r['config']==name and r['ms']==ms];n=len(selected);hits=sum(r['hit'] for r in selected)
   z=1.959964;q=hits/n;den=1+z*z/n;center=(q+z*z/(2*n))/den;radius=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
   summary.append({'config':name,'ms':ms,'hits':hits,'positions':n,'wilson_95_lower':center-radius,'wilson_95_upper':center+radius})
   save(a.out,{'engine_sha256':sha(a.engine),'model_sha256':sha(MODEL),'fixture_sha256':sha(fixture),
     'metric':'exact published first reply, not exhaustive correctness; historical model training provenance unknown',
     'summary':summary,'results':results})
   print(summary[-1],flush=True)
 with a.out.with_suffix('.csv').open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
if __name__=='__main__':main()
