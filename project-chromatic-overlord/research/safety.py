#!/usr/bin/env python3
"""Separate Linux sanitizer builds; run only AFTER timed experiments finish."""
import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--build-root',type=Path,required=True);p.add_argument('--completed',type=Path,required=True);a=p.parse_args()
    assert platform.system()=='Linux';assert json.loads(a.completed.read_text())['completed'];a.out.mkdir(parents=True,exist_ok=True)
    settings={'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1','TSAN_OPTIONS':'halt_on_error=1'}
    report={'os':platform.platform(),'sanitizer_options':settings,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'cpp_sha256':{str(x.relative_to(ROOT)):sha(x) for d in ['src','include'] for x in sorted((ROOT/d).rglob('*')) if x.is_file()},'builds':[],'passed':False}
    measured=json.loads((a.completed.parent/'manifest.json').read_text());assert report['cpp_sha256']==measured['cpp_sha256']
    report['measured_engine_sha256']=measured['engine_sha256'];report['same_cpp_as_measurement']=True
    for kind,option in [('asan','CHROMATIC_ASAN'),('tsan','CHROMATIC_TSAN')]:
        build=a.build_root/kind;assert not build.exists(),'Use a fresh sanitizer build directory'
        record={'kind':kind,'steps':[]};report['builds'].append(record)
        for name,command in [('configure',['cmake','-S',str(ROOT),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=RelWithDebInfo',f'-D{option}=ON']),
            ('build',['cmake','--build',str(build),'--parallel','2']),('ctest',['ctest','--test-dir',str(build),'--output-on-failure'])]:
            path=a.out/f'{kind}-{name}.log';start=time.monotonic()
            with path.open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**settings})
            record['steps'].append({'name':name,'command':command,'exit_code':r.returncode,'seconds':time.monotonic()-start,'log_sha256':sha(path)})
            (a.out/'safety.json').write_text(json.dumps(report,indent=2)+'\n')
            if r.returncode:break
        exe=build/'chromatic_overlord'
        if exe.exists():record['engine_sha256']=sha(exe)
        record['passed']=len(record['steps'])==3 and all(x['exit_code']==0 for x in record['steps'])
    report['passed']=all(x['passed'] for x in report['builds']);(a.out/'safety.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'builds':[{k:x[k] for k in ['kind','passed']} for x in report['builds']]}))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
