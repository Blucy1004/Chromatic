import chess, subprocess, sys
from pathlib import Path
exe = Path(sys.argv[1]).resolve()
model = Path(sys.argv[2]).resolve()
positions = [
  ("startpos", "startpos"),
  ("before-Nh5", "fen rnbqkb1r/pp2pppp/2p2n2/2N5/3P4/2P5/PP2BPPP/R1BQK1NR b KQkq - 2 7"),
  ("before-Qxg3", "fen r1b4r/pp3kpp/n1p1p3/8/P2P3q/2P1B1N1/1P3PPP/R2QK1NR b KQ - 2 16"),
  ("after-hxg3", "fen r1b4r/pp3kpp/n1p1p3/8/P2P4/2P1B1P1/1P3PP1/R2QK1NR b KQ - 0 17"),
]
eng = subprocess.Popen([str(exe), "--model", str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1)
def send(s):
    eng.stdin.write(s+"\n"); eng.stdin.flush()
def until(pred):
    lines=[]
    while True:
        line=eng.stdout.readline()
        if not line: raise RuntimeError("engine exited before expected UCI response")
        lines.append(line.rstrip())
        if pred(line.rstrip()): return lines
send("uci"); uci=until(lambda x:x=="uciok")
send("isready"); ready=until(lambda x:x=="readyok")
rows=[]
for mode,rrn in [("PVS",False),("PVS",True),("MCTS",False)]:
    send(f"setoption name SearchMode value {mode}")
    send(f"setoption name UseRRN value {'true' if rrn else 'false'}")
    send("isready"); ready=until(lambda x:x=="readyok")
    for label,pos in positions:
        send("position "+pos)
        send("go nodes 512" if mode=="MCTS" else "go depth 3")
        lines=until(lambda x:x.startswith("bestmove "))
        bm=next(x.split()[1] for x in lines if x.startswith("bestmove "))
        board=chess.Board() if pos=="startpos" else chess.Board(" ".join(pos.split()[1:]))
        if bm!="0000":
            move=chess.Move.from_uci(bm)
            if move not in board.legal_moves: raise RuntimeError(f"illegal move {bm} for {label}/{mode}/rrn={rrn}")
        info=[x for x in lines if x.startswith("info depth ") or x.startswith("info nodes ")]
        rrnline=[x for x in lines if "RRN ordering enabled" in x]
        rows.append(f"{mode} RRN={rrn} {label}: {info[-1] if info else '(no info)'}; {bm}; {rrnline[-1] if rrnline else ''}")
send("quit"); eng.wait(timeout=10)
print("UCI_ID="+uci[0]); print("READY="+" | ".join(x for x in ready if "RRN" in x or "warning" in x)); print("\n".join(rows)); print("exit=",eng.returncode)
