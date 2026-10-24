import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import re


class Native:
    def __init__(self, executable, model=None, timeout=60):
        self.timeout = timeout
        self.process = subprocess.Popen([str(Path(executable).resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', bufsize=1)
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.options=set()
        self.name='unknown'
        try:
            self.send('uci')
            def handshake(line):
                if line.startswith('option name '):self.options.add(line[12:].split(' type ',1)[0])
                if line.startswith('id name '):self.name=line[8:]
                return line=='uciok'
            self.until(handshake)
            if model:
                if 'ModelFile' not in self.options:raise ValueError('engine does not support ModelFile')
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

    def until(self, predicate, timeout=None, errors=None):
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
                if errors is None:raise RuntimeError(line)
                errors.append(line)
            if predicate(line):
                return line

    def ready(self):
        self.send('isready')
        errors=[]
        self.until(lambda s: s == 'readyok',errors=errors)
        if errors:raise RuntimeError('; '.join(errors))

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

    def configure(self, options):
        for name,value in options.items():
            if name not in self.options:raise ValueError('unsupported UCI option: '+name)
            if isinstance(value,bool):value=str(value).lower()
            self.send(f'setoption name {name} value {value}')
        self.ready()

    def pin_cpu(self,mask=1):
        import os
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            function=kernel.SetProcessAffinityMask
            function.argtypes=[wintypes.HANDLE,ctypes.c_size_t];function.restype=wintypes.BOOL
            if not function(int(self.process._handle),mask):raise ctypes.WinError(ctypes.get_last_error())
        elif hasattr(os,'sched_setaffinity'):
            os.sched_setaffinity(self.process.pid,{i for i in range(mask.bit_length()) if mask&(1<<i)})
        else:raise RuntimeError('CPU affinity is unavailable')

    def play(self, board, *, milliseconds=None, nodes=None):
        if (milliseconds is None)==(nodes is None):raise ValueError('choose one UCI search budget')
        self.position(board)
        self.send(f'go movetime {milliseconds}' if milliseconds is not None else f'go nodes {nodes}')
        info={};start=time.perf_counter()
        def response(line):
            if line.startswith('info '):
                for field in ('nodes','time','nps'):
                    match=re.search(r'\b'+field+r' (\d+)',line)
                    if match:info[field]=int(match[1])
            return line.startswith('bestmove ')
        errors=[]
        line=self.until(response,timeout=max(5,(milliseconds or 0)/1000+3),errors=errors)
        if errors:raise RuntimeError('; '.join(errors))
        return {'bestmove':line.split()[1],'elapsed_seconds':time.perf_counter()-start,'info':info}

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
