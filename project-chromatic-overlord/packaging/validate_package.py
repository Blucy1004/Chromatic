#!/usr/bin/env python3
"""Native tests against extracted package bytes, with no repository model paths."""
import argparse,hashlib,json,os,platform,queue,re,struct,subprocess,threading,time,xml.etree.ElementTree as ET
from pathlib import Path
import chess
CHAMPION='f6a00902c23e7f32f96b028ac36464cc9012cd8774d5e6b0dee5e1c94e1c03c9'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Engine:
    def __init__(self,exe,cwd,basename_argv0=False):
        self.p=subprocess.Popen([exe.name if basename_argv0 else str(exe)],executable=str(exe),cwd=cwd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='strict')
        self.q=queue.Queue();self.thread=threading.Thread(target=self.reader,daemon=True);self.thread.start()
    def reader(self):
        try:
            for line in self.p.stdout:self.q.put(line.rstrip('\r\n'))
        finally:self.q.put(None)
    def send(self,line):self.p.stdin.write(line+'\n');self.p.stdin.flush()
    def until(self,predicate,timeout=5):
        lines=[];end=time.monotonic()+timeout
        while time.monotonic()<end:
            item=self.q.get(timeout=max(.01,end-time.monotonic()))
            if item is None:raise RuntimeError('Engine closed before expected response')
            lines.append(item)
            if predicate(item):return lines
        raise TimeoutError('UCI response timeout')
    def set(self,name,value):self.send(f'setoption name {name} value {str(value).lower()}')
    def search(self,fen=chess.STARTING_FEN,limit='movetime 100'):
        self.send('position fen '+fen);start=time.monotonic();self.send('go '+limit)
        lines=self.until(lambda s:s.startswith('bestmove '),max(5,int(limit.split()[-1])/1000+2) if limit.startswith('movetime') else 10)
        detail=next(json.loads(s.removeprefix('info string overlord ')) for s in lines if s.startswith('info string overlord '))
        move=lines[-1].split()[1]
        if move!='0000':assert chess.Move.from_uci(move) in chess.Board(fen).legal_moves
        return detail,1000*(time.monotonic()-start)
    def close(self):
        if self.p.poll() is None:self.send('quit')
        self.p.wait(timeout=5);self.thread.join(timeout=2)
        assert self.p.returncode==0,self.p.stderr.read()
    def __enter__(self):return self
    def __exit__(self,*args):
        if args[0] is None:self.close()
        elif self.p.poll() is None:self.p.kill();self.p.wait()
def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--imports',type=Path,required=True);p.add_argument('--embedded-manifest',type=Path,required=True);a=p.parse_args()
    assert platform.system()=='Windows' and struct.calcsize('P')==8,'Native Windows x64 required'
    package=a.package.resolve();exe=package/'chromatic.exe';model=package/'default.zero';a.out.mkdir(parents=True,exist_ok=True)
    outside=a.out/'unrelated working directory';outside.mkdir(exist_ok=True)
    checks=[];report={'native_windows':True,'os':platform.platform(),'engine_sha256':sha(exe),'model_sha256':sha(model),'package_path':str(package),'checks':checks,'passed':False}
    def record(name,**data):checks.append({'name':name,'passed':True,**data});(a.out/'package.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    raw=exe.read_bytes();offset=struct.unpack_from('<I',raw,0x3c)[0];assert struct.unpack_from('<H',raw,offset+4)[0]==0x8664
    assert sha(model)==CHAMPION;record('PE_x64_and_original_champion')
    xml=ET.parse(a.embedded_manifest);cp=xml.find('.//{http://schemas.microsoft.com/SMI/2019/WindowsSettings}activeCodePage');assert cp is not None and cp.text=='UTF-8';record('actual_embedded_UTF8_manifest')
    imports=re.findall(r'^\s+([A-Za-z0-9_.-]+\.dll)\s*$',a.imports.read_text(encoding='utf-8-sig'),re.M|re.I)
    assert imports and not any(re.match(r'(?:msvcp|vcruntime|concrt|vcomp|ucrtbased|clang_rt)',x,re.I) for x in imports),imports
    allowed={'kernel32.dll','advapi32.dll','user32.dll','shell32.dll','ole32.dll','oleaut32.dll','ws2_32.dll','bcrypt.dll','ntdll.dll','ucrtbase.dll'}
    assert all(x.lower() in allowed or x.lower().startswith(('api-ms-win-','ext-ms-win-')) for x in imports),imports
    record('only_windows_system_DLL_imports',imports=imports)
    assert ' ' in str(package) and "'" in str(package) and any(ord(c)>127 for c in str(package))
    with Engine(exe,outside) as e:
        e.send('uci');lines=e.until(lambda s:s=='uciok')
        for name in ['CounterexampleVerify','AdaptiveBudget','ForcingSearch','TacticalVerify','ResearchMetrics']:assert f'option name {name} type check default false' in lines
        assert 'option name Threads type spin default 1 min 1 max 32' in lines
        e.send('zero model');loaded=json.loads(e.until(lambda s:s.startswith('{'))[-1]);assert loaded['loaded'] and loaded['sha256']==CHAMPION
        assert Path(loaded['path']).resolve()==model
        e.send('isready');e.until(lambda s:s=='readyok');record('automatic_model_from_unicode_path_and_unrelated_CWD',loaded=loaded,experimental_defaults_off=True)
        for ms in [100,1000,10000]:
            r,elapsed=e.search(limit=f'movetime {ms}');assert r['research'] is None and elapsed<ms+1500
            record(f'go_{ms}ms',requested_ms=ms,elapsed_ms=elapsed,bestmove=r['bestmove'])
        e.set('ResearchMetrics',True);e.set('MateProofNodes',0)
        counters={'CounterexampleVerify':'counterexample_calls','AdaptiveBudget':'adaptive_calls','ForcingSearch':'forcing_checks','TacticalVerify':'material_calls'}
        for name,counter in counters.items():
            for enabled in [True,False]:
                e.set(name,enabled);r,_=e.search(limit='nodes 64');assert bool(r['research'][counter])==enabled,(name,counter,r)
                record(name+'_'+str(enabled),calls=r['research'][counter])
        for threads in [2,1]:
            e.set('Threads',threads);r,_=e.search(limit='nodes 64');assert r['simulations']==64
            if threads==2:assert r['peak_inflight']==2
            record('shared_tree_threads_'+str(threads),peak_inflight=r['peak_inflight'])
        e.set('Threads',2);e.set('MaxNodes',16);e.set('TreeMemoryMB',1);e.set('SearchTelemetry',True)
        r,_=e.search(limit='nodes 1024');assert r['simulations']==1024 and r['prune_count']>0
        assert all(x['root_statistics_preserved'] for x in r['telemetry']['prunes']);record('reclamation_preserves_root_statistics',prunes=r['prune_count'],completed_visits=r['simulations'])
        for name,value in {'MaxNodes':100000,'TreeMemoryMB':64,'SearchTelemetry':False,'ResearchMetrics':False,'MateProofNodes':256,'Threads':1}.items():e.set(name,value)
        for side,fen in [('white','7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'),('black','8/8/8/8/8/6k1/5q2/7K b - - 0 1')]:
            r,elapsed=e.search(fen);board=chess.Board(fen);board.push_uci(r['bestmove']);assert board.is_checkmate();record('real_tactic_mate1_'+side,bestmove=r['bestmove'],elapsed_ms=elapsed)
        for name,fen in [('stalemate','7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'),('checkmate','7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'),('fifty_move','4k3/8/8/8/8/8/R7/4K3 w - - 100 70')]:
            r,_=e.search(fen);assert r['bestmove']=='0000' and r['terminal'];record(name)
        e.send('position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8');e.send('go nodes 16')
        assert e.until(lambda s:s.startswith('bestmove '))[-1]=='bestmove 0000';record('threefold_history')
        for name in counters:e.set(name,True)
        e.set('Threads',2);e.send('position startpos');e.send('go infinite');time.sleep(.1);start=time.monotonic();e.send('stop');e.until(lambda s:s.startswith('bestmove '),3)
        record('stop_all_options_parallel',latency_ms=1000*(time.monotonic()-start))
        e.send('position startpos');e.send('go infinite');time.sleep(.05);e.send('quit');e.p.wait(timeout=3);assert e.p.returncode==0;record('quit_during_search',exit_code=0)
    for depth,expected in [(3,8902),(4,197281),(5,4865609)]:
        run=subprocess.run([str(exe),'--perft',str(depth)],cwd=outside,text=True,capture_output=True,check=True,timeout=60);assert int(run.stdout.strip())==expected
        record('perft_'+str(depth),nodes=expected)
    with Engine(exe,outside,basename_argv0=True) as e:
        e.send('zero model');loaded=json.loads(e.until(lambda s:s.startswith('{'))[-1])
        assert loaded['loaded'] and loaded['sha256']==CHAMPION and Path(loaded['path']).resolve()==model
        record('basename_argv0_GUI_style_model_discovery',loaded=loaded)
    with Engine(exe,outside) as e:
        e.send('\ufeffzero model');loaded=json.loads(e.until(lambda s:s.startswith('{'))[-1]);assert loaded['loaded'] and loaded['sha256']==CHAMPION
        record('UTF8_BOM_first_command',loaded_sha256=loaded['sha256'])
    comspec=os.environ.get('COMSPEC',r'C:\Windows\System32\cmd.exe')
    def batch(path,stdin=None):
        command=f'"{comspec}" /d /s /c ""{path}""'
        return subprocess.run(command,cwd=outside,input=stdin,text=True,encoding='utf-8',capture_output=True,timeout=30)
    console=batch(package/'run-console.cmd','uci\nisready\nzero model\nquit\n');assert console.returncode==0,console.stderr
    assert 'uciok' in console.stdout and 'readyok' in console.stdout
    loaded=next(json.loads(s) for s in console.stdout.splitlines() if s.startswith('{'));assert loaded['loaded'] and loaded['sha256']==CHAMPION;record('double_click_console_launcher',exit_code=0)
    verify=batch(package/'self-check.cmd');assert verify.returncode==0 and 'PASS:' in verify.stdout,(verify.stdout,verify.stderr);record('self_check_launcher_and_file_hashes',exit_code=0)
    report['passed']=True;(a.out/'package.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'passed':True,'checks':len(checks)}))
if __name__=='__main__':main()
