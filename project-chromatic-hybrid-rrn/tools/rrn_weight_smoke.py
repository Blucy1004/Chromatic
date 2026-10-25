import subprocess,sys
from pathlib import Path
import chess
exe=Path(sys.argv[1]).resolve(); weights=Path(sys.argv[2]).resolve(); fen='r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5'
p=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',bufsize=1)
def send(s):p.stdin.write(s+'\n');p.stdin.flush()
def read_until(pred):
 out=[]
 while True:
  line=p.stdout.readline()
  if not line:raise RuntimeError('engine exited')
  out.append(line.rstrip())
  if pred(line.rstrip()):return out
send('uci');uci=read_until(lambda s:s=='uciok')
send('setoption name SearchMode value PVS');send('setoption name UseRRN value true');send('setoption name RRNWeights value '+str(weights));send('isready');ready=read_until(lambda s:s=='readyok')
send('position fen '+fen);send('go depth 3');search=read_until(lambda s:s.startswith('bestmove '))
move=next(x.split()[1] for x in search if x.startswith('bestmove '))
if chess.Move.from_uci(move) not in chess.Board(fen).legal_moves:raise RuntimeError('illegal move '+move)
meta=[x for x in search if 'RRN ordering enabled' in x]
if not meta or 'learned=true' not in meta[-1]:raise RuntimeError('trained metadata not reflected: '+repr(meta))
send('quit');p.wait(timeout=10)
print('\n'.join(ready));print('\n'.join(meta));print('bestmove',move,'legal=true');print('last_info',next((x for x in reversed(search) if x.startswith('info ')),'none'))

