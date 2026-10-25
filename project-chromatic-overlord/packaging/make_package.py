#!/usr/bin/env python3
"""Assemble a new distribution without modifying any existing deployment/model."""
import argparse,hashlib,json,shutil,struct,subprocess,zipfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];REPO=PROJECT.parent
CHAMPION_SHA='f6a00902c23e7f32f96b028ac36464cc9012cd8774d5e6b0dee5e1c94e1c03c9'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def manifest(package,validated=False):
    info=json.loads((package/'BUILD_INFO.json').read_text(encoding='utf-8'))
    files={p.relative_to(package).as_posix():sha(p) for p in sorted(package.rglob('*')) if p.is_file() and p.name not in ['MANIFEST.json','SHA256SUMS.txt']}
    save(package/'MANIFEST.json',{**info,'native_package_validated':validated,'files':files})
    all_files={**files,'MANIFEST.json':sha(package/'MANIFEST.json')}
    (package/'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(all_files.items())),encoding='utf-8')
def archive(package,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package.parent).as_posix())
def prepare(a):
    assert not a.out.exists(),'Choose a fresh package directory'
    raw=a.engine.read_bytes();offset=struct.unpack_from('<I',raw,0x3c)[0]
    assert raw[:2]==b'MZ' and raw[offset:offset+4]==b'PE\0\0' and struct.unpack_from('<H',raw,offset+4)[0]==0x8664,'Expected native Windows x64 PE'
    assert sha(a.model)==CHAMPION_SHA,'Only the preserved champion is bundled; no promotion'
    a.out.mkdir(parents=True);shutil.copy2(a.engine,a.out/'chromatic.exe');shutil.copy2(a.model,a.out/'default.zero')
    for name in ['run-console.cmd','self-check.cmd','rebuild.cmd','verify-package.ps1']:
        content=(PROJECT/'packaging'/name).read_text(encoding='utf-8')
        (a.out/name).write_bytes(('\r\n'.join(content.splitlines())+'\r\n').encode('utf-8'))
    for name in ['LICENSE','THIRD_PARTY.md']:shutil.copy2(REPO/'project-chromatic'/name,a.out/name)
    source=a.out/'source';source.mkdir()
    shutil.copy2(PROJECT/'CMakeLists.txt',source/'CMakeLists.txt')
    shutil.copy2(REPO/'project-chromatic/LICENSE',source/'LICENSE')
    for directory in ['src','include','tests','packaging']:
        for p in (PROJECT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                target=source/p.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    info={'name':'CHROMATIC Overlord experimental','version':'0.1.0','build':'Release','architecture':'Windows x64','runtime':'MSVC /MT','active_code_page':'UTF-8',
        'source_commit':commit,'source_url':f'https://github.com/Blucy1004/Chromatic/tree/{commit}/project-chromatic-overlord',
        'engine_sha256':sha(a.out/'chromatic.exe'),'model':{'file':'default.zero','sha256':CHAMPION_SHA,'format':'ZERO0001','inputs':1280,'width':64,'blocks':2,'actions':4672,'parameters':394179,'teacher_positions':128,'training_steps':200},
        'defaults':{'Threads':1,'SharedTree':True,'MateProofNodes':256,**{name:False for name in ['CounterexampleVerify','AdaptiveBudget','ForcingSearch','TacticalVerify','ResearchMetrics']}},
        'champion_promoted':False,'existing_deployment_replaced':False}
    save(a.out/'BUILD_INFO.json',info)
    readme='''# CHROMATIC Windows x64 Release

## 실행

1. ZIP 전체를 한 폴더에 압축 해제하세요.
2. 체스 GUI에서는 UCI 엔진으로 `chromatic.exe`를 등록하세요. 모델은 옆의 `default.zero`에서 자동으로 로딩됩니다.
3. 콘솔로 실행하려면 `run-console.cmd`를 더블클릭하세요.

이 패키지는 UCI 체스 엔진입니다. 체스판 GUI는 포함되어 있지 않습니다.
Windows 10 1903+ / Windows 11 x64용이며, 네이티브 실행 검증은 Windows Server 2022 x64에서 수행합니다.
실행에 Python, CMake, Visual Studio 또는 별도 Visual C++ Redistributable 설치는 필요하지 않습니다.

콘솔에서 다음 명령을 차례로 입력하세요.

```
uci
isready
position startpos
go movetime 1000
```

`bestmove` 응답 후 `quit`로 종료합니다. 탐색 중에는 `stop`을 사용할 수 있습니다.
`self-check.cmd`는 패키지 파일 해시와 실제 로딩된 모델을 확인합니다.

## 포함된 엔진과 모델

Operation Overlord 실험 엔진의 Release 빌드입니다. 기존 초기 챔피언(교사128개·200스텝)을 그대로 포함했습니다.
CounterexampleVerify / AdaptiveBudget / ForcingSearch / TacticalVerify / ResearchMetrics는 기본 OFF, Threads는1입니다.
이 패키지 작성은 실험 기능의 기본 활성화나 후보 모델 승격을 의미하지 않습니다.

`MANIFEST.json`과 `SHA256SUMS.txt`에 파일별 SHA-256, 모델과 실행 파일의 출처가 있습니다.
네이티브 패키지 검사 결과는 `validation` 폴더에 포함됩니다. 실제 대국 Elo 개선은 미확정입니다.

## 소스와 재빌드

정확한 대응 C++ 소스, 테스트, 빌드 설정은 `source`에 포함됩니다. GPL-3.0-or-later, 라이선스는 `LICENSE`를 확인하세요.
재빌드가 필요한 경우에만 Visual Studio2022 Desktop C++ 및 CMake3.20+를 설치하고 `rebuild.cmd`를 실행하세요.
원래 배포본·챔피언 파일은 변경하지 않았습니다.
'''
    (a.out/'README.md').write_text(readme,encoding='utf-8');(a.out/'README.txt').write_text(readme,encoding='utf-8-sig')
    manifest(a.out);archive(a.out,a.zip);print(json.dumps({'prepared':str(a.out),'zip':str(a.zip),'engine_sha256':info['engine_sha256']}))
def finalize(a):
    report=json.loads((a.validation/'package.json').read_text(encoding='utf-8'))
    info=json.loads((a.package/'BUILD_INFO.json').read_text(encoding='utf-8'))
    assert report['passed'] and report['native_windows'] and report['engine_sha256']==sha(a.package/'chromatic.exe')==info['engine_sha256']
    assert report['model_sha256']==sha(a.package/'default.zero')==CHAMPION_SHA
    assert '100% tests passed, 0 tests failed out of 11' in (a.validation/'ctest.log').read_text(encoding='utf-8-sig')
    target=a.package/'validation';assert not target.exists();shutil.copytree(a.validation,target)
    manifest(a.package,True);archive(a.package,a.zip)
    a.zip.with_suffix('.zip.sha256').write_text(sha(a.zip)+'  '+a.zip.name+'\n',encoding='utf-8')
    print(json.dumps({'packaged':str(a.zip),'bytes':a.zip.stat().st_size,'sha256':sha(a.zip)}))
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--engine',type=Path,required=True);prep.add_argument('--model',type=Path,required=True);prep.add_argument('--out',type=Path,required=True);prep.add_argument('--zip',type=Path,required=True)
    final=sub.add_parser('finalize');final.add_argument('--package',type=Path,required=True);final.add_argument('--validation',type=Path,required=True);final.add_argument('--zip',type=Path,required=True)
    a=p.parse_args();(prepare if a.command=='prepare' else finalize)(a)
if __name__=='__main__':main()
