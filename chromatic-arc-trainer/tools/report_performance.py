"""Regenerate PERFORMANCE.md and charts from existing measured logs only.

No PyTorch import, device initialization, training, or download is performed.
Repeat --runs/--metadata for a final merged dataset or additional benchmark groups.
Matplotlib is optional; use the report environment or --no-charts.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
NUMERIC=['wall_seconds','positions_per_second','pure_training_seconds','training_samples',
    'training_samples_per_second','transfer_seconds','data_processing_seconds','validation_seconds','checkpoint_seconds','recovery_seconds',
    'rss_peak_bytes','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_utilization_mean','cpu_process_percent_mean',
    'io_read_bytes','io_write_bytes','policy_validation_loss','wdl_validation_loss','policy_test_loss','wdl_test_loss','global_step']

def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):return isinstance(value,(int,float)) and math.isfinite(value)
def stats(values):
    values=[float(v) for v in values if finite(v)]
    return {'n':len(values),'median':statistics.median(values),'min':min(values),'max':max(values)} if values else {'n':0,'median':None,'min':None,'max':None}
def number(value,digits=2):return f'{value:,.{digits}f}' if finite(value) else 'NA'
def mib(value):return number(value/1024**2,1) if finite(value) else 'NA'
def relative(path,root=ROOT):
    path=Path(path)
    try:return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:return os.path.relpath(path,root).replace('\\','/')
def link(path,label,root=ROOT):return f'[{label}]({relative(path,root)})'
def locate_run(row,root):
    original=Path(row.get('directory',''))
    if original.is_dir():return original
    normalized=str(row.get('directory','')).replace('\\','/')
    marker='/chromatic-arc-trainer/'
    return root/normalized.split(marker,1)[-1] if marker in normalized else original
def condition(row):
    return (row.get('implementation','arc'),int(row.get('positions',0)),row.get('device','cpu'),row.get('precision','fp32'),
        int(row.get('threads',1)),int(row.get('batch_size',32)),row.get('cache_mode','auto'),row.get('validation_mode','full'),bool(row.get('profile',False)))
def condname(key):
    impl,n,device,precision,threads,batch,cache,validation,profile=key
    name=f'CPU {threads}T' if device=='cpu' else f'Arc {precision.upper()}'
    return name+(f' B{batch}' if batch!=32 else '')+(f' / {cache}' if cache not in ['auto','none'] else '')

STAGES=[('압축 읽기·Zstandard·spool',['source_read','source_spool_read','source_spool_write','zstd_decompress']),
    ('JSONL 파싱',['json_parse']),('FEN 검증',['fen_validation']),('PV·합법수 검증',['pv_validation']),
    ('중복 제거·균형 샘플링',['dedup_sampling']),('Policy/WDL 라벨 변환',['label_conversion']),
    ('샤드 직렬화·역직렬화',['shard_serialize','shard_deserialize']),
    ('binary 인코딩·PyTorch 배치 로딩',['binary_cache_prepare','binary_encode','data_loading']),
    ('CPU→XPU·상주 batch gather',['host_to_device','device_batch_gather']),
    ('Forward/Backward/Optimizer·안전 검사',['forward','backward','optimizer','gradient_clip','finite_check','precision_reference']),
    ('Validation',['validation']),('Checkpoint 저장·정리',['checkpoint_save','checkpoint_housekeeping'])]

def load_runs(paths):
    runs={};inputs=[];ignored=[]
    for path in paths:
        path=Path(path)
        if not path.is_file():continue
        text=path.read_text(encoding='utf-8');lines=text.splitlines()
        for index,line in enumerate(lines):
            if not line.strip():continue
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1 and not text.endswith('\n'):ignored.append(str(path));continue
                raise
            row=dict(row,_input_file=str(path.resolve()))
            identity=(row.get('directory'),row.get('key'),row.get('validation_mode','full'))
            runs[identity]=row
        inputs.append({'path':str(path.resolve()),'sha256':sha(path),'rows_present':len(lines)})
    return list(runs.values()),inputs,ignored

def grouped(rows):
    groups=defaultdict(list)
    for row in rows:groups[condition(row)].append(row)
    output=[]
    for key,runs in sorted(groups.items(),key=lambda x:(x[0][1],x[0][7],x[0][8],x[0][0],x[0][2],x[0][4],x[0][5],x[0][3])):
        good=[r for r in runs if r.get('status')=='ok']
        item={'condition':list(key),'label':condname(key),'recorded_runs':len(runs),'completed_runs':len(good),
            'statuses':dict(Counter(r.get('status','unknown') for r in runs)),
            'keys':[r.get('key') for r in runs],
            'holdout_digests':sorted({r['holdout_digest'] for r in good if r.get('holdout_digest')}),
            'metrics':{metric:stats([r.get(metric) for r in good]) for metric in NUMERIC}}
        output.append(item)
    return output

def median(group,key):return group['metrics'][key]['median']
def peak(group,key):return group['metrics'][key]['max']
def series(groups,validation='full',positions=10000):
    found=[g for g in groups if g['condition'][0]=='arc' and g['condition'][1]==positions and g['condition'][5]==32 and g['condition'][6]=='auto'
        and g['condition'][7]==validation and not g['condition'][8] and g['completed_runs']]
    return sorted(found,key=lambda g:(g['condition'][2]!='cpu',g['condition'][4],g['condition'][3]!='fp32'))

def profile_rows(rows,root,output):
    result=[]
    for row in rows:
        if row.get('status')!='ok' or not row.get('profile',False):continue
        source=locate_run(row,root)/'candidate/profile/profile.json'
        safe=re.sub('[^A-Za-z0-9_.-]','-',str(row.get('key','profile'))+'-'+row.get('validation_mode','full'))
        snapshot=output/'profiles'/(safe+'.json')
        profile=read(source) if source.is_file() else read(snapshot)
        if not profile:continue
        if source.is_file():save(snapshot,profile)
        stages=profile['stages'];values=[]
        for label,names in STAGES:
            selected=[stages[n]['seconds'] for n in names if n in stages]
            value=sum(selected) if selected else (0. if label.startswith('CPU→') and row.get('device')=='cpu' else None)
            values.append(value)
        result.append({'key':row['key'],'validation_mode':row.get('validation_mode','full'),'device':row['device'],
            'threads':row.get('threads',1),'wall_seconds':row.get('wall_seconds'),'profile':profile,'category_seconds':values,
            'source_profile':str(source),'copied_profile':str(snapshot),
            'checkpoint_recovery_seconds':stages.get('checkpoint_recovery',{}).get('seconds')})
    return sorted(result,key=lambda x:(x['validation_mode']!='full',x['device']!='cpu',x['key']))

def charts(root,groups,profiles,no_charts=False):
    if no_charts:return {'created':False,'reason':'--no-charts'}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:return {'created':False,'reason':str(exc)}
    directory=root/'reports/graphs';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    full=series(groups,'full');selected=series(groups,'selected')
    baseline_group=next((g for g in groups if g['condition'][0]=='baseline' and g['condition'][1]==10000 and g['completed_runs']),None)
    baseline=median(baseline_group,'wall_seconds') if baseline_group else None
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,data,title in [(axes[0],full,'Full PV validation'),(axes[1],selected,'Selected-analysis validation / separate experiment')]:
        x=np.arange(len(data));vals=[median(g,'wall_seconds') for g in data]
        colors=['#548d7d' if g['condition'][2]=='cpu' else '#4f80b8' if g['condition'][3]=='fp32' else '#c08c42' for g in data]
        ax.bar(x,vals,color=colors,alpha=.9)
        for i,(g,value) in enumerate(zip(data,vals)):
            st=g['metrics']['wall_seconds'];ax.errorbar(i,value,yerr=[[value-st['min']],[st['max']-value]],fmt='none',ecolor='#333333',capsize=4)
            ax.text(i,value+2,f'{value:.1f}s\nn={g["completed_runs"]}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(x,[g['label'].replace('Arc ','Arc\n') for g in data],rotation=20,ha='right');ax.set_ylabel('External wall seconds (lower is better)')
        ax.set_title(title);ax.set_ylim(0,max([150]+[v*1.2 for v in vals]));ax.grid(axis='y',alpha=.2)
        if baseline:
            b=baseline_group['metrics']['wall_seconds'];ax.axhspan(b['min'],b['max'],color='#777777',alpha=.06)
            ax.axhline(baseline,color='#666666',linestyle='--',label=f'Old trainer median: {baseline:.1f}s (n={b["n"]})')
        ax.axhline(132.6,color='#aaaaaa',linestyle=':',label='Historical Python 3.12 result: 132.6s');ax.legend(fontsize=8)
    fig.suptitle('10,000 positions / 3 epochs / batch 32 / identical fixed validation',fontsize=14)
    fig.savefig(directory/'pipeline-performance.png');plt.close(fig)

    if profiles:
        english=['Input + Zstandard + spool','JSON parsing','FEN validation','PV legality','Dedup + sampling','Policy/WDL conversion',
            'Shard serialize / read','Binary encode + data loading','Host-to-XPU / gather','Train + clip + finite','Validation','Checkpoint + housekeeping']
        fig,ax=plt.subplots(figsize=(13,7),layout='constrained');y=np.arange(len(STAGES));width=.8/max(len(profiles),1)
        colors=['#7eab9e','#7d9fc9','#326c57','#1f5e9b','#c08c42','#9d6c9c']
        for i,p in enumerate(profiles):
            vals=[v if finite(v) else 0 for v in p['category_seconds']]
            label=('CPU 1T' if p['device']=='cpu' else 'Arc FP32')+' / '+p['validation_mode']
            ax.barh(y+(i-(len(profiles)-1)/2)*width,vals,height=width,color=colors[i%len(colors)],label=label)
        ax.set_yticks(y,english);ax.invert_yaxis();ax.set_xlabel('Exclusive stage wall seconds (measured profiling runs)')
        ax.set_title('Representative stage profiles / GPU spans synchronized');ax.legend(ncols=2,fontsize=8);ax.grid(axis='x',alpha=.2)
        fig.savefig(directory/'stage-profiles.png');plt.close(fig)

    data=full+selected
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained');x=np.arange(len(data))
    labels=[g['label']+('\nselected' if g['condition'][7]=='selected' else '\nfull') for g in data]
    for ax,key,title,unit,use_peak in [(axes[0,0],'rss_peak_bytes','Observed RAM samples / separate measurement scopes','MiB',True),
        (axes[0,1],'gpu_peak_allocated_bytes','XPU peak tensor allocation / not dedicated VRAM','MiB',True),
        (axes[1,0],'positions_per_second','Completed positions per external wall second','Positions/sec',False),
        (axes[1,1],'training_samples_per_second','Epoch + Replay samples per pure training second','Training samples/sec',False)]:
        vals=[peak(g,key) if use_peak else median(g,key) for g in data]
        if unit=='MiB':vals=[v/1024**2 if finite(v) else float('nan') for v in vals]
        else:vals=[v if finite(v) else float('nan') for v in vals]
        ax.bar(x,vals,color=['#548d7d' if g['condition'][2]=='cpu' else '#4f80b8' for g in data]);ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=8)
        ax.set_title(title,fontsize=11);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Distinct throughput denominators; shared GPU memory must not be added to process RAM',fontsize=13)
    fig.savefig(directory/'resources-throughput.png');plt.close(fig)
    return {'created':True,'files':['reports/graphs/pipeline-performance.png','reports/graphs/stage-profiles.png','reports/graphs/resources-throughput.png']}

def generate(root,runs_paths,metadata_paths,no_charts=False):
    rows,inputs,ignored=load_runs(runs_paths);groups=grouped(rows);output=root/'reports/performance';output.mkdir(parents=True,exist_ok=True)
    metadata=[{'path':str(p),'sha256':sha(p),'content':read(p)} for p in metadata_paths if Path(p).is_file()]
    profiles=profile_rows(rows,root,output);graph=charts(root,groups,profiles,no_charts)
    recovery_file=root/'reports/checkpoint-recovery-timings.csv';recovery_rows=[]
    if recovery_file.is_file():
        with recovery_file.open(encoding='utf-8',newline='') as stream:recovery_rows=list(csv.DictReader(stream))
    recovery_groups={device:stats([float(r['checkpoint_recovery_exclusive_seconds']) for r in recovery_rows if r['device']==device]) for device in ['cpu','xpu']}
    save(output/'summary.json',{'format':'chromatic-arc-performance-report-v1','generated_utc':datetime.now(timezone.utc).isoformat(),
        'inputs':inputs,'metadata':metadata,'ignored_unfinished_last_lines':ignored,'recorded_runs':len(rows),
        'status_counts':dict(Counter(r.get('status') for r in rows)),'groups':groups,'profiles':profiles,'charts':graph,
        'separate_checkpoint_recovery':recovery_groups})
    fields=['implementation','positions','device','precision','cpu_threads','batch_size','cache_mode','validation_mode','profile','recorded','completed','statuses']
    fields +=[name+'_'+part for name in NUMERIC for part in ['n','median','min','max']]
    with (output/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for g in groups:
            d=dict(zip(fields[:9],g['condition']));d.update(recorded=g['recorded_runs'],completed=g['completed_runs'],statuses=canonical_json(g['statuses']))
            for name,st in g['metrics'].items():d.update({name+'_'+key:value for key,value in st.items()})
            writer.writerow(d)
    with (output/'runs-derived.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    baseline={g['condition'][1]:g for g in groups if g['condition'][0]=='baseline' and g['completed_runs']}
    reference=median(baseline[10000],'wall_seconds') if 10000 in baseline else None
    full=series(groups,'full');selected=series(groups,'selected')
    best=min(selected,key=lambda g:median(g,'wall_seconds')) if selected else None
    env=read(root/'reports/environment.json',{});quality=read(root/'reports/quality-selected/validation.json',{})
    current_validation='full'
    source=(root/'shardtrain/run.py').read_text(encoding='utf-8')
    match=re.search(r"add_argument\('--validation-mode'.*?default='([^']+)'",source)
    if match:current_validation=match.group(1)
    lines=['# CHROMATIC ARC — 전체 학습 처리량과 검증 결과','',
        '이 보고서는 이미 저장된 실행 로그만 읽어 생성한다. 장치 연산이나 추가 학습·다운로드를 실행하지 않는다. 실제 Arc XPU FP32/BF16 학습은 동작했지만 이 작은 MLP·batch 32에서는 GPU가 전체 처리량의 최선이라는 결과는 나오지 않았다. 반복 FEN/합법수 인코딩 캐시와 데이터 검증 병목이 중요했고, CPU 스레드 수를 함께 비교했다.','',
        f'생성 시점에 기록된 파이프라인 실행은 {len(rows)}개다. 상태는 '+', '.join(f'{k} {v}개' for k,v in Counter(r.get('status') for r in rows).items())+'다. 표의 n은 성공 완료한 실제 반복 수이며, 오류·deadline 중단은 중앙값에 넣지 않는다. 진행 중인 그룹은 나중에 같은 생성기로 다시 집계할 수 있다.','',
        '## 환경과 비교 계약','',
        'Windows x64 / Python 3.11.9 / PyTorch 2.14.1+xpu / NumPy 2.4.6, Intel Arc 140T 통합 GPU와 기존 width 64·blocks 2 MLP를 사용했다. 입력 1,280·Policy 4,672·WDL 3 구조와 챔피언 가중치는 유지했다. GPU의 표시 16GB는 공유 시스템 메모리를 포함할 수 있으며 독립 전용 VRAM으로 해석하지 않는다. '+link(root/'reports/environment.json','환경 원시 기록',root), '',
        '주요 10,000개 조건은 같은 챔피언에서 독립 시작, 3에폭, seed 2026, shard 1,000, batch 32, LR 0.001, Replay 256, 32업데이트마다 저장, 관리 디스크 1GiB·메모리 4GiB다. validation/test는 동일한 고정 1,000개를 사용한다. 입력은 로컬 공식 CC0 60,000행 fixture이며 다운로드 시간과 원격 다운로드는 없다. 새 ledger·spool·binary cache로 시작했지만 OS 파일 캐시·온도·다른 프로세스 부하는 완전히 통제하지 않았다. full과 selected 검증 그룹은 별도 시점의 실행이다.','',
        '전체 시간은 외부 wall clock이며 Python 시작·데이터 준비·학습·독립 validation·내보내기·체크포인트 저장을 포함한다. 순수 학습 시간은 forward/backward/Optimizer·gradient clip·finite 검사 구간이다. XPU 전송·학습 경계는 synchronize로 실제 완료를 잰다. 체크포인트 경로의 Trainer process_seconds는 시작 범위가 다르므로 외부 wall과 혼용하지 않는다.','',
        '## 보존한 과거 기록과 같은 환경의 재현','',
        '|포지션|보존된 Python 3.12 CPU1 기록|현재 Python 3.11 재현 중앙값 [범위] 초|현재 재현 n|', '|---:|---:|---:|---:|']
    for n,historical in [(100,15.4),(1000,24.8),(10000,132.6)]:
        group=baseline.get(n);st=group['metrics']['wall_seconds'] if group else None
        lines.append(f'|{n:,}|{historical:.1f}초|{number(st["median"])} [{number(st["min"])}–{number(st["max"])}]|{group["completed_runs"]}|' if group else f'|{n:,}|{historical:.1f}초|NA|0|')
    lines +=['','두 기준선은 Python·패키지 환경과 실행 시점이 다르다. 초기 재현 105.29초와 개선된 process-tree monitor로 재실행한 82.25초를 모두 보존해 10,000개 기준은 두 값의 중앙값으로 사용했다. 빠른 한 번이나 느린 한 번만 골라 기준으로 삼지 않았다. 구현 최적화 가속 배율은 현재 환경의 기존 학습기 재현을 기준으로 따로 계산하며, 과거 132.6초와의 단순 차이를 모두 새 코드 효과라고 주장하지 않는다. 재현이 1회인 항목은 3회 중앙값이라고 표시하지 않는다.','']
    if best and reference:
        t=median(best,'wall_seconds')
        lines +=[f'**현재 조건의 가장 빠른 검증된 selected 설정은 {best["label"]}, 3회 중앙값 {number(t)}초다. 같은 Python 환경의 기존 학습기 중앙값 {number(reference)}초 대비 {number(reference/t)}×다.** 과거 132.6초 대비 {number(132.6/t)}×는 환경이 다른 참고 비교이며 주 가속 배율과 구분한다.','']
    lines +=[
        '## 10,000개 전체 파이프라인 — full PV 검증','',
        '|조건|성공 n|전체 중앙값 [최소–최대] 초|순수 학습 초|전송 초|Positions/s|Training samples/s|재현 기준 가속|Policy / WDL val|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in full:
        w=g['metrics']['wall_seconds'];speed=reference/w['median'] if reference else None
        lines.append(f'|{g["label"]}|{g["completed_runs"]}|{number(w["median"])} [{number(w["min"])}–{number(w["max"])}]|{number(median(g,"pure_training_seconds"))}|{number(median(g,"transfer_seconds"),3)}|{number(median(g,"positions_per_second"),1)}|{number(median(g,"training_samples_per_second"),0)}|{number(speed)}×|{number(median(g,"policy_validation_loss"),6)} / {number(median(g,"wdl_validation_loss"),6)}|')
    samples=sorted({int(r['training_samples']) for r in rows if r.get('status')=='ok' and r.get('positions')==10000 and r.get('batch_size')==32 and finite(r.get('training_samples'))})
    sample_text=', '.join(f'{n:,}' for n in samples) if samples else 'NA'
    updates=sorted({int(r['global_step']) for r in rows if r.get('status')=='ok' and r.get('positions')==10000 and r.get('batch_size')==32 and finite(r.get('global_step'))})
    update_text=', '.join(f'{n:,}' for n in updates) if updates else 'NA'
    lines +=['',f'완료된 반복이 2회인 조건은 n=2의 중앙값이다. timeout 반복을 완료로 간주하거나 임의 보간하지 않았다. 위 batch 32 실행은 같은 신규 포지션 10,000개를 3에폭 학습하며, Replay를 합친 실제 학습 샘플은 **{sample_text}개**, Optimizer 단계는 **{update_text}개**였다. **Positions/sec는 완료 포지션/전체 시간**, **Training samples/sec는 Epoch+Replay 샘플/순수 학습 시간**이므로 서로 다른 분모·분자다.','',
        '## 선택된 평가부터 검증하는 데이터 경로 — 별도 실험','']
    if quality:
        lines +=[f'공식 fixture {quality.get("records",0):,}행을 full과 selected 정규화로 비교했다. 수용 행은 {quality.get("accepted",{}).get("full",0):,}개이며 의미 비교 mismatch는 {quality.get("mismatch_count","NA")}개였다. 선택된 평가·정책·WDL·confidence·FEN/history·분할 identity·provenance의 canonical 결과를 비교했고, 명시적으로 달라진 낮은 평가의 rejection-audit 범위는 제외했다. '+link(root/'reports/quality-selected/validation.json','독립 정규화 감사',root),'',
            f'정규화 구성요소의 시간은 full {number(quality.get("normalization_seconds",{}).get("full"))}초, selected {number(quality.get("normalization_seconds",{}).get("selected"))}초로 {number(quality.get("selected_speedup"),3)}×였다. 이는 압축 해제·JSON 파싱·학습·저장을 제외한 **구성요소 가속**이며 전체 학습 가속으로 바꾸어 쓰지 않는다. selected도 실제 사용한 PV 전체의 합법성을 검증하며, 사용하지 않는 낮은 분석의 세부 거부 목록까지 같은 범위로 수집하는 모드는 아니다.','']
    lines +=['|조건|성공 n|전체 중앙값 [최소–최대] 초|순수 학습 초|전송 초|재현 기준 가속|Policy / WDL val|', '|---|---:|---:|---:|---:|---:|---:|']
    for g in selected:
        w=g['metrics']['wall_seconds'];lines.append(f'|{g["label"]}|{g["completed_runs"]}|{number(w["median"])} [{number(w["min"])}–{number(w["max"])}]|{number(median(g,"pure_training_seconds"))}|{number(median(g,"transfer_seconds"),3)}|{number(reference/w["median"] if reference else None)}×|{number(median(g,"policy_validation_loss"),6)} / {number(median(g,"wdl_validation_loss"),6)}|')
    lines +=['',f'보고서 생성 시 코드의 기본 validation mode는 **{current_validation}**다. 표본 전체의 의미 일치와 별도 파이프라인 결과를 근거로 판단하되, 한 소규모 fixture의 결과를 모든 미래 데이터에 대한 증명으로 해석하지 않는다. full/selected 실행 시점과 측정 driver 변경도 있으므로 전체 시간 차이를 전부 한 함수의 효과로 단정하지 않는다.','',
        '![전체 처리 시간 비교](reports/graphs/pipeline-performance.png)','',
        '## RAM·GPU·CPU·I/O 관측','',
        '|검증/조건|관측 RAM 최고 MiB|XPU 할당 최고 MiB|XPU 예약 최고 MiB|CPU process 평균 %|GPU busiest engine 평균 %|읽기 / 쓰기 MiB|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in full+selected:
        lines.append(f'|{g["condition"][7]} / {g["label"]}|{mib(peak(g,"rss_peak_bytes"))}|{mib(peak(g,"gpu_peak_allocated_bytes"))}|{mib(peak(g,"gpu_peak_reserved_bytes"))}|{number(median(g,"cpu_process_percent_mean"),1)}|{number(median(g,"gpu_utilization_mean"),1)}|{mib(median(g,"io_read_bytes"))} / {mib(median(g,"io_write_bytes"))}|')
    lines +=['','RAM은 연속 모든 순간을 계측한 물리 최고점이 아니라 표본·allocator의 관측값이다. 초기 Windows venv python.exe 외부 sampler가 실제 interpreter 대신 redirector를 측정한 문제가 발견됐다. 해당 원시 로그·wall time은 보존하고, ARC 자원은 실제 프로세스 내부 resource-summary로 보정한 combined-corrected를 사용했다. 기존 v1 기준선은 커밋 RSS 표본만 회수할 수 있어 CPU/I/O는 NA다. 이후 selected driver는 process-tree와 실제 내부 자원을 함께 기록한다. 이 측정 범위 차이를 작은 RAM 차이의 개선 근거로 사용하지 않는다. '+link(root/'reports/benchmarks/resource-correction.json','보정 근거',root),'',
        'GPU%는 현재 PID의 가장 바쁜 Windows GPU engine이며 EU occupancy가 아니다. 유효한 관측값이 없으면 NA와 이유를 보존했고 0%로 대체하지 않았다. CPU process%는 여러 코어에서 100%를 넘을 수 있다. I/O는 프로세스 카운터이며 물리 드라이브 전체 트래픽이 아니다. 통합 GPU의 allocator와 프로세스 RSS는 겹칠 수 있으므로 합산하지 않는다.','',
        '![자원과 서로 다른 처리량](reports/graphs/resources-throughput.png)','',
        '## 12단계 대표 프로파일','',
        '다음은 별도의 Profile 실행에서 얻은 exclusive wall seconds다. GPU stage는 전후 synchronize를 사용한다. CPU source worker는 GPU를 동기화하지 않는다. 부모 inclusive stage와 child를 중복 합산하지 않으며 다른 스레드의 동시 실행 합계는 wall time보다 커질 수 있다. Profile-off인 나머지 반복의 세부 단계는 NA이며 대표값을 복사해 채우지 않았다.','']
    labels=[('CPU1' if p['device']=='cpu' else 'Arc FP32')+' '+p['validation_mode'] for p in profiles]
    lines +=['|단계|'+'|'.join(labels)+'|','|---|'+'|'.join(['---:']*len(profiles))+'|'] if profiles else ['아직 완료된 Profile 실행이 없다.']
    for i,(label,names) in enumerate(STAGES):
        if profiles:lines.append('|'+label+'|'+'|'.join(number(p['category_seconds'][i],3) for p in profiles)+'|')
    if profiles:
        lines +=['','학습 구간 안의 세부 초는 다음과 같다. 위 10번째 행의 부분 항목이므로 다시 더하지 않는다.','',
            '|학습 부분|'+'|'.join(labels)+'|','|---|'+'|'.join(['---:']*len(profiles))+'|']
        for name in ['forward','backward','optimizer','gradient_clip','finite_check']:
            lines.append('|'+name+'|'+'|'.join(number(p['profile']['stages'].get(name,{}).get('seconds'),3) for p in profiles)+'|')
    lines +=['','복구 단계는 새 실행에서 호출하지 않으므로 위 저장 시간에 섞지 않는다. `checkpoint_recovery`는 별도 실제 Resume Profile의 다음 표로 보고한다. 원본 decode/JSON과 선택 평가·PV 비용, 캐시 생성/배치 로딩을 분리했다. 표에 없는 data_preparation exclusive bookkeeping, Python 시작, 오프닝 앵커 준비, 장치 초기화와 계측 자체 비용도 전체 wall에는 포함될 수 있어 12행 합계가 전체 시간과 일치할 필요는 없다.','']
    if recovery_rows:
        lines +=['|별도 selected Resume 장치|실제 복구 span n|중앙값 초|최소–최대 초|','|---|---:|---:|---:|']
        for device,st in recovery_groups.items():lines.append(f'|{device}|{st["n"]}|{number(st["median"],6)}|{number(st["min"],6)}–{number(st["max"],6)}|')
        lines +=['','완료 재개·세 강제 종료·미러 및 이전 checkpoint 복구의 실제 load/검증 구간이다. Python/장치 초기화와 남은 학습·고정 validation을 포함한 재개 프로세스 전체 시간은 수초 이상이며 이 0.0x초와 다르다. '+link(recovery_file,'각 복구 Profile·SHA·전체 재개 시간 CSV',root),'']
    lines +=[
        '|Profile 조건|Profile 전체 초|동일 조건 정상 중앙값 초|관측 비율 차이|Checkpoint recovery 초|', '|---|---:|---:|---:|---:|']
    for p,label in zip(profiles,labels):
        normal=next((g for g in full+selected if g['condition'][2]==p['device'] and g['condition'][4]==p['threads'] and g['condition'][7]==p['validation_mode'] and g['condition'][3]=='fp32'),None)
        wall=median(normal,'wall_seconds') if normal else None;delta=(p['wall_seconds']/wall-1)*100 if wall else None
        lines.append(f'|{label}|{number(p["wall_seconds"])}|{number(wall)}|{number(delta,1)}%|{number(p["checkpoint_recovery_seconds"],3)}|')
    lines +=['','이 비율 차이는 별도 실행의 관측값으로 순수 profiler 오버헤드만 분리한 통제 실험은 아니다. Profile은 동기화와 많은 이벤트 기록을 추가하므로 주요 처리량은 Profile-off 반복을 기준으로 한다.','',
        '![대표 단계별 시간](reports/graphs/stage-profiles.png)','',
        '## 작은 데이터와 배치 크기 해석','',
        '|포지션 / 조건|batch|성공 n|전체 초|순수 학습 초|Optimizer 단계|Policy / WDL val|', '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['condition'][0]!='arc' or g['condition'][1] not in [100,1000] or g['condition'][8] or not g['completed_runs']:continue
        lines.append(f'|{g["condition"][1]:,} / {g["label"]}|{g["condition"][5]}|{g["completed_runs"]}|{number(median(g,"wall_seconds"))}|{number(median(g,"pure_training_seconds"),3)}|{number(median(g,"global_step"),0)}|{number(median(g,"policy_validation_loss"),6)} / {number(median(g,"wdl_validation_loss"),6)}|')
    lines +=['','배치 32/64/128/256/512와 BF16·상주·사전 할당·누적·Optimizer의 이미 인코딩된 배치 행렬은 '+link(root/'docs/MICROBENCHMARK.md','MICROBENCHMARK',root)+'와 '+link(root/'reports/benchmarks/microbenchmark.json','원시 57조건',root)+'에 있다. 큰 배치는 같은 3에폭에서도 Optimizer 업데이트 수를 바꾸므로, 작은 데이터의 loss/속도만 보고 batch 확대를 품질 보존 최적화로 채택하지 않았다. 상주 미니배치 마이크로벤치마크와 전체 샤드 상주는 서로 다른 메모리 실험이다.','',
        '## 수치 문제를 발견하고 제외한 경로','',
        '실제 Arc 환경에서 기본 foreach gradient clip이 norm을 약 2.163705로 반환했지만 CPU float64 기준은 약 1.901946이었다. gradient 자체의 최대 오차는 약 1.27e−7이어서 clip 경로의 문제를 분리할 수 있었다. XPU clip을 foreach=False로 고정한 뒤 norm·clipped gradient·한 단계 업데이트를 다시 검증했다. 수정 전 자료는 진단용 pre-clip-fix에 보존하고 정규 결과로 사용하지 않았다. '+link(root/'reports/clip-diagnostic.json','clip 진단',root)+' / '+link(root/'reports/benchmarks/measurement-code.json','측정 코드 근거',root),'']
    numeric=read(root/'reports/numeric-xpu-fp32.json',{})
    if numeric:
        errors=numeric.get('errors',{});lines +=[f'수정 후 FP32 CPU/XPU 오차: Policy logits {number_scientific(errors.get("policy_logits"))}, WDL 확률 {number_scientific(errors.get("wdl_probability"))}, gradient {number_scientific(errors.get("gradient"))}, 업데이트 가중치 {number_scientific(errors.get("updated_weights"))}. 판정은 '+str(numeric.get('passed'))+'이며, 이는 같은 입력 한 배치의 수치 검사다. 긴 학습의 모든 가중치가 비트 단위로 같다는 뜻은 아니다. '+link(root/'reports/numeric-xpu-fp32.json','FP32 원시 검증',root),'']
    lines +=['BF16은 FP32 master weights와 FP32 loss/softmax를 유지하고 비정상 손실/gradient 또는 사전 정의된 loss 악화 시 Optimizer 전에 FP32로 복귀한다. 짧은 BF16 통과만으로 기본 정밀도를 변경하지 않았다. fused/foreach Optimizer, gradient accumulation, 사전 할당 전송의 효과는 장치·배치에 따라 달랐고 일관된 전체 파이프라인 근거 없이 기본값으로 채택하지 않았다.','',
        '전체 샤드 XPU 상주 모드는 4GiB 프로세스 한도에서 bad allocation을 만나 두 번 실패했다. 실패한 실행의 중간 처리량은 완료 처리량으로 사용하지 않았고 해당 샤드·체크포인트를 보존했다. 6GiB 한도의 추가 상주 실험은 별도 조건으로 다음에 표시한다. 메모리 한도와 검증 경로가 다르고 1회뿐이므로 기본 copy/auto 경로보다 공정하게 우수하다는 근거로 쓰지 않는다. 완료되지 않은 실행을 CPU로 조용히 이어 붙이지 않았다.','',
        '|상주 실험 검증 모드|메모리 한도 GiB|완료 / 시도|완료 wall 중앙값 초|관측 RAM 최고 MiB|XPU 할당 최고 MiB|', '|---|---:|---:|---:|---:|---:|']
    resident=defaultdict(list)
    for row in rows:
        if row.get('cache_mode')=='device':resident[(row.get('validation_mode','full'),row.get('max_memory_gb',4))].append(row)
    for (mode,cap),cases in sorted(resident.items()):
        ok=[r for r in cases if r.get('status')=='ok'];wall=stats([r.get('wall_seconds') for r in ok])
        ram=stats([r.get('rss_peak_bytes') for r in cases]);gpu=stats([r.get('gpu_peak_allocated_bytes') for r in cases])
        lines.append(f'|{mode}|{cap}|{len(ok)} / {len(cases)}|{number(wall["median"])}|{mib(ram["max"])}|{mib(gpu["max"])}|')
    lines +=['', '`torch.compile`은 Windows XPU 전체가 미지원인 것은 아니지만 현재 환경의 격리 시험에서 cp949/UnicodeDecodeError 및 종료 상한 문제가 발생해 활성화하지 않았다. '+link(root/'reports/benchmarks/compile-probe.json','격리 compile 기록',root),'',
        '|미완료 조건|상태|반복|관측 종료 시간 초|원시 로그|', '|---|---|---:|---:|---|']
    for row in rows:
        if row.get('status')=='ok':continue
        logfile=Path(row['stdout']) if row.get('stdout') else None
        lines.append(f'|{row.get("name")}|{row.get("status")}|{row.get("repeat")}|{number(row.get("wall_seconds"))}|{link(logfile,"로그",root) if logfile else "NA"}|')
    lines +=['','## 최종 정확성·복구·기존 상태 보존','']
    if numeric:
        lines +=[f'혼합 입력 32개에서 Policy와 WDL 양쪽의 유효 라벨·gradient·업데이트를 실제로 확인했다. 양쪽 헤드 gradient 검사 판정은 {numeric.get("both_heads_gradients_exercised",False)}이며 FP32 한 단계 가중치 최대 오차는 {number_scientific(numeric.get("errors",{}).get("updated_weights"))}였다. 예전 Value 가중치 0인 입력만으로 두 헤드 검사를 통과했다고 판단하지 않는다.','']
    native=read(root/'reports/native-summary.json',{})
    if native:lines +=[f'독립 C++ SIMD/sparse-policy 추론은 CPU4-selected·XPU-FP32-selected·XPU-BF16 후보 각 10개, 총 **{native.get("cases",0)}개**가 통과했다. 동일 PT에서 내보낸 FP32 `.zero`와 PyTorch를 비교했고 엔진·챔피언은 수정하지 않았다. '+link(root/'reports/native-summary.json','C++ 추론 전체 결과',root),'']
    lines +=['selected CPU/XPU 각각 5개 실제 중단·손상 복구 시나리오에서 모델·Optimizer·Scheduler·모든 RNG·Replay·진행 상태가 중단 없는 같은 장치 기준과 정확히 같았다. selected와 full의 기준 내보내기 SHA도 같은 장치에서 일치했다. '+link(root/'reports/recovery-selected-summary.json','selected 복구 원시 요약',root),'']
    imported=read(root/'reports/legacy-import-validation.json',{})
    if imported:lines +=[f'기존 완료 100개 실행을 원본 읽기 전용으로 복사해 새 ARC head로 이행했다. step {imported.get("global_step_before")}→{imported.get("global_step_after")}, 신규 학습 업데이트 {imported.get("new_training_updates")}회, 의미 상태 {len(imported.get("identical_semantic_state_keys",[]))}개와 `.zero` 일치를 확인했다. '+link(root/'reports/legacy-import-validation.json','실제 기존 실행 import',root),'']
    unitlog=root/'reports/unit-tests-final.log'
    if unitlog.is_file():
        unittext=unitlog.read_text(encoding='utf-8');unitmatch=re.search(r'(\d+) passed, (\d+) skipped, (\d+) subtests passed',unittext)
        if unitmatch:lines +=[f'Python 3.11 최종 일반 검사 {unitmatch.group(1)}개와 {unitmatch.group(3)}개 subtest가 통과했다. 일반 실행에서 생략한 {unitmatch.group(2)}개 선택 실행 시험은 별도 실제 장치·복구 검증과 구분한다. '+link(unitlog,'검사 로그',root),'']
    lines +=['## 채택 범위와 남은 한계','',
        f'현재 기본은 Device auto, CpuThreads 1, BatchSize 32, FP32, cache auto, Prefetch 0, 기본 AdamW, accumulation 1이다. validation mode는 {current_validation}다. selected는 의미 비교·고정 검증·동일 장치 실제 복구가 확인된 범위에서 채택했다. mode가 없던 이전 ARC Resume는 full로 복원해 이전 의미·서명을 보존한다. CPU 스레드나 배치·정밀도·상주 모드를 측정 한 번만으로 자동 확대하지 않는다.','',
        'auto는 등록된 전체 파이프라인 추천을 먼저 검토한다. 하드웨어/모델 구현·초기 가중치·source·규모·Epochs·seed·Replay·샤드·업데이트 수·배치·스레드·검증 모드 제약과 품질 gate가 맞아야 사용한다. 현재 등록 결과는 CPU4/B32가 전체 처리량 추천이고, 같은 CPU1 제약에서 CPU와 XPU를 3회씩 비교한 auto 장치는 CPU다. 제약이 다르면 이 속도를 외삽하지 않고 마이크로벤치마크/짧은 실측으로 돌아간다. '+link(root/'reports/benchmarks/pipeline-recommendation.json','실측 등록·제약·품질 gate',root),'',
        '고정 holdout·confidence 라벨·Replay·성공 샤드 삭제 조건과 기존 `.zero` 형식을 유지한다. 기존 CPU/XPU 체크포인트 이동·같은 장치 강제 종료/Resume는 별도 '+link(root/'docs/VALIDATION.md','검증 계약',root)+' 및 '+link(root/'reports/recovery-validation-summary.json','복구 결과',root)+'로 확인한다. 기존 전체 학습 상태를 원본 보존하며 복사하는 경로는 '+link(root/'docs/LEGACY_IMPORT.md','Legacy import',root)+'다. 엔진·Muyeong·챔피언·이전 실험 자료를 덮어쓰지 않는다.','',
        '이번 결과는 100·1,000·10,000개 로컬 데이터 실험이다. 100,000개 학습이나 대규모 다운로드는 실행하지 않았다. 학습 loss·속도 개선은 Elo나 체스 기력의 증명이 아니며 새 모델은 후보로만 저장한다. 데이터 처리·저장·안전 검사 비용 때문에 미니배치 peak samples/sec를 전체 처리량으로 외삽할 수 없다.','',
        '## 원시 자료와 재생성','',
        '- '+link(output/'summary.json','모든 조건·반복·stage JSON',root)+' / '+link(output/'summary.csv','집계 CSV',root)+' / '+link(output/'runs-derived.jsonl','출처 경로가 붙은 파생 JSONL',root),
        '- '+link(root/'reports/benchmarks/all-runs.jsonl','최종 44개 합본 JSONL',root)+' / '+link(root/'reports/benchmarks/all-metadata.json','전체 그룹 메타데이터',root),
        '- '+link(root/'reports/benchmarks/combined-corrected.jsonl','초기 전체 그룹의 자원 보정 JSONL',root)+' / '+link(root/'reports/benchmarks/pipeline-arc-final/plan.json','초기 전체 계획',root),
        '- '+link(root/'reports/benchmarks/pipeline-arc-selected/pipeline-runs.jsonl','selected 그룹 원시 JSONL',root)+' / '+link(root/'reports/benchmarks/pipeline-arc-selected/plan.json','selected 계획',root),
        '- 각 실행의 command.json, training.stdout.log, candidate/logs/training.csv·JSONL, candidate/resources·profile 아래 원시 기록',
        '- '+link(root/'reports/quality-selected/validation.json','60,000행 의미 일치 감사',root)+' / '+link(root/'reports/benchmarks/microbenchmark.json','인코딩 배치 행렬',root), '',
        '최종 합본이 준비되면 다음과 같이 같은 보고서를 다시 생성한다. 이 명령은 로그 읽기와 표·그래프 생성만 하며 PyTorch/XPU를 import하지 않는다. Matplotlib이 없는 환경에서는 --no-charts를 지정할 수 있다.','',
        '```powershell', r'python .\tools\report_performance.py `',
        r'  --runs .\reports\benchmarks\all-runs.jsonl `',
        r'  --metadata .\reports\benchmarks\all-metadata.json', '```','']
    (root/'PERFORMANCE.md').write_text('\n'.join(lines),encoding='utf-8')
    return {'runs':len(rows),'groups':len(groups),'profiles':len(profiles),'graphs':graph,'report':str(root/'PERFORMANCE.md')}

def canonical_json(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def number_scientific(value):return f'{value:.3e}' if finite(value) else 'NA'
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,action='append',help='Existing JSONL; repeat for additional groups or final merged rows')
    parser.add_argument('--metadata',type=Path,action='append',default=[])
    parser.add_argument('--output-root',type=Path,default=ROOT)
    parser.add_argument('--no-charts',action='store_true')
    args=parser.parse_args(argv);root=args.output_root.resolve()
    final=root/'reports/benchmarks/all-runs.jsonl'
    paths=args.runs or ([final] if final.is_file() else [root/'reports/benchmarks/combined-corrected.jsonl',root/'reports/benchmarks/pipeline-arc-selected/pipeline-runs.jsonl'])
    metadata=args.metadata or ([root/'reports/benchmarks/all-metadata.json'] if (root/'reports/benchmarks/all-metadata.json').is_file() else [root/'reports/benchmarks/pipeline-arc-final/plan.json',root/'reports/benchmarks/pipeline-arc-selected/plan.json',
        root/'reports/benchmarks/measurement-code.json',root/'reports/benchmarks/resource-correction.json'])
    print(json.dumps(generate(root,paths,metadata,args.no_charts),ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
