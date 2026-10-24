import json
from pathlib import Path
import queue
import subprocess
import threading
import time


class Native:
    def __init__(self, executable, model=None, timeout=60):
        self.timeout = timeout
        self.process = subprocess.Popen([str(Path(executable).resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', bufsize=1)
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.send('uci')
            self.until(lambda s: s == 'uciok')
            if model:
                self.send('setoption name ModelFile value ' + str(Path(model).resolve()))
            self.ready()
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def send(self, text):
        self.process.stdin.write(text + '\n')
        self.process.stdin.flush()

    def until(self, predicate, timeout=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('native engine response timed out')
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty as exc:
                raise TimeoutError('native engine response timed out') from exc
            if line is None:
                raise RuntimeError('native engine exited: ' + str(self.process.poll()))
            if line.startswith('info string error '):
                raise RuntimeError(line)
            if predicate(line):
                return line

    def ready(self):
        self.send('isready')
        self.until(lambda s: s == 'readyok')

    def position(self, board):
        root = board.root().fen(en_passant='fen')
        moves = ' '.join(m.uci() for m in board.move_stack)
        self.send('position fen ' + root + (' moves ' + moves if moves else ''))

    def command(self, text):
        self.send(text)
        result = json.loads(self.until(lambda s: s.startswith('{')))
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def search(self, simulations=128, seed=1, noise=0.25, temperature=1.0):
        return self.command(f'zero search {simulations} {seed} {noise} {temperature}')

    def close(self):
        if self.process.poll() is None:
            try:
                self.send('quit')
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
