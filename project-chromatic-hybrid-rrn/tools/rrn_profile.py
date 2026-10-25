import json
import subprocess
import sys
import time
from pathlib import Path

import chess

exe = Path(sys.argv[1]).resolve()
model = Path(sys.argv[2]).resolve()
weights = Path(sys.argv[3]).resolve()
out = Path(sys.argv[4]).resolve()
positions = [
    ("startpos", "startpos"),
    ("before-Nh5", "fen rnbqkb1r/pp2pppp/2p2n2/2N5/3P4/2P5/PP2BPPP/R1BQK1NR b KQkq - 2 7"),
    ("before-Qxg3", "fen r1b4r/pp3kpp/n1p1p3/8/P2P3q/2P1B1N1/1P3PPP/R2QK1NR b KQ - 2 16"),
    ("after-hxg3", "fen r1b4r/pp3kpp/n1p1p3/8/P2P4/2P1B1P1/1P3PP1/R2QK1NR b KQ - 0 17"),
]
engine = subprocess.Popen([str(exe), "--model", str(model)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", bufsize=1)

def send(s):
    engine.stdin.write(s + "\n")
    engine.stdin.flush()

def until(prefix):
    lines = []
    start = time.monotonic()
    while time.monotonic() - start < 90:
        line = engine.stdout.readline()
        if not line:
            raise RuntimeError("engine exited before response")
        lines.append(line.rstrip())
        if line.startswith(prefix):
            return lines
    raise TimeoutError(f"waiting for {prefix}")

send("uci")
uci = until("uciok")
supports_min_depth = any("option name RRNMinDepth " in line for line in uci)
send("setoption name RRNWeights value " + str(weights))
send("isready")
ready = until("readyok")
rows = []
for use_rrn in (False, True):
    send("setoption name SearchMode value PVS")
    send("setoption name UseRRN value " + ("true" if use_rrn else "false"))
    if supports_min_depth:
        send("setoption name RRNMinDepth value 4")
    send("setoption name PVSHashMB value 32")
    send("setoption name MateProofNodes value 0")
    send("isready")
    until("readyok")
    for label, pos in positions:
        for limit in ("nodes 50000", "movetime 500"):
            send("position " + pos)
            send("go " + limit)
            lines = until("bestmove ")
            info = next((x for x in reversed(lines) if x.startswith("info depth ")), "")
            best = next(x for x in lines if x.startswith("bestmove ")).split()[1]
            board = chess.Board() if pos == "startpos" else chess.Board(" ".join(pos.split()[1:]))
            if best != "0000" and chess.Move.from_uci(best) not in board.legal_moves:
                raise RuntimeError(f"illegal bestmove {best} in {label}")
            rows.append({"rrn": use_rrn, "position": label, "limit": limit,
                         "info": info, "bestmove": best,
                         "rrn_messages": [x for x in lines if "RRN" in x or "learned=" in x]})
send("quit")
engine.wait(timeout=10)
out.write_text(json.dumps({"uci": uci, "ready": ready, "rows": rows,
                           "exit_code": engine.returncode}, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
