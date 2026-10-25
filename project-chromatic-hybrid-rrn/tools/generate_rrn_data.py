import argparse, json, random, subprocess, time
from pathlib import Path
import chess

class Uci:
  def __init__(self,exe):
    self.p=subprocess.Popen([exe],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',bufsize=1)
    self._send('uci'); lines=[]
    while True:
      line=self.p.stdout.readline()
      if not line: raise RuntimeError('Stockfish ended during UCI handshake')
      lines.append(line.strip())
      if line.strip()=='uciok':break
    self.id=next((x for x in lines if x.startswith('id name ')),'id name unknown')
    self._send('setoption name Threads value 1');self._send('setoption name Hash value 16');self._send('isready')
    while self.p.stdout.readline().strip()!='readyok':pass
  def _send(self,s):self.p.stdin.write(s+'\n');self.p.stdin.flush()
  def best(self,fen,nodes):
    self._send('position fen '+fen);self._send(f'go nodes {nodes}')
    best=None;score=None;searched=0
    while True:
      line=self.p.stdout.readline()
      if not line: raise RuntimeError('Stockfish ended during search')
      parts=line.strip().split()
      if parts and parts[0]=='info':
        if 'nodes' in parts:
          try:searched=int(parts[parts.index('nodes')+1])
          except:pass
        if 'score' in parts:
          try:score=' '.join(parts[parts.index('score')+1:parts.index('pv') if 'pv' in parts else len(parts)])
          except:pass
      if parts and parts[0]=='bestmove':best=parts[1];break
    return best,score,searched
  def close(self):
    try:self._send('quit');self.p.wait(timeout=5)
    except:self.p.kill()

p=argparse.ArgumentParser(); p.add_argument('--stockfish',required=True); p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=20261025); a=p.parse_args()
rng=random.Random(a.seed); total_games=32; positions_per_game=24
engine=Uci(a.stockfish); rows=[]; seen=set(); starts=time.time()
print('teacher',engine.id,'threads=1 hash=16 nodes=1024',flush=True)
try:
  for game in range(total_games):
    attempt=0
    while True:
      game_seed=a.seed+game*1000+attempt; grng=random.Random(game_seed); board=chess.Board(); trajectory=[]
      for ply in range(8+positions_per_game):
        if board.is_game_over(claim_draw=True): break
        if 8 <= ply < 8+positions_per_game:
          fen=board.fen(en_passant='legal'); key=' '.join(fen.split()[:4]); trajectory.append((fen,key))
        board.push(grng.choice(list(board.legal_moves)))
      if len(trajectory)==positions_per_game: break
      attempt+=1
      if attempt>100: raise RuntimeError(f'could not generate full trajectory for game index {game}')
    split='train' if game<24 else 'heldout'
    for j,(fen,key) in enumerate(trajectory):
      if key in seen: continue
      seen.add(key); best,score,nodes=engine.best(fen,1024);b=chess.Board(fen);move=chess.Move.from_uci(best)
      if move not in b.legal_moves: continue
      rows.append({'split':split,'trajectory_seed':game_seed,'sample':j,'fen':fen,'bestmove':best,'teacher_nodes':nodes,'teacher_score':score})
    print(f'game {game+1}/{total_games}: seed={game_seed} split={split} samples={len(trajectory)} total={len(rows)} elapsed={time.time()-starts:.1f}s',flush=True)
finally:engine.close()
train=[x for x in rows if x['split']=='train']; held=[x for x in rows if x['split']=='heldout']
if len(train)<400 or len(held)<128: raise RuntimeError(f'insufficient unique samples: train={len(train)} heldout={len(held)}')
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
payload={'source':'random-legal-trajectories; game-level split','seed':a.seed,'games':total_games,'positions_per_game':positions_per_game,'teacher':engine.id+'; Threads=1; Hash=16 MB; nodes=1024','train':train,'heldout':held}
out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
print(f'WROTE {out} train={len(train)} heldout={len(held)} elapsed={time.time()-starts:.1f}s')
