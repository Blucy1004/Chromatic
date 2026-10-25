#!/usr/bin/env python3
"""Render the six requested reports from retained analysis, without fitting/selection."""
import argparse,csv,itertools,json,math,statistics
from collections import defaultdict
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
OUTPUT=PROJECT
MODELS=['zero128','lichess100','lichess1000','lichess10000'];CORE=['zero128','lichess10000'];BITS='CAFTP';SYSTEMS=['Linux','Windows']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(p.open()))
def label(m):return ''.join(c for i,c in enumerate(BITS) if int(m)>>i&1) or 'base'
def num(x,d=1):return '미측정' if x is None or x=='' else f'{float(x):.{d}f}'
def interval(r,key='delta_pp',lo='lo95_pp',hi='hi95_pp'):
    return f"{num(r[key])} [{num(r[lo])}, {num(r[hi])}]"
def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in data])+'\n'
def write(name,text):(OUTPUT/name).write_text(text.strip()+'\n')
def main():
    global OUTPUT
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--reports-out',type=Path,default=PROJECT);a=p.parse_args();d=a.analysis;OUTPUT=a.reports_out;OUTPUT.mkdir(parents=True,exist_ok=True)
    plan=read(a.plan);matrix=rows(d/'matrix.csv');contrasts=rows(d/'paired-contrasts.csv');shapley=rows(d/'shapley.csv');interactions=rows(d/'interactions.csv');model_effects=rows(d/'model-effects.csv')
    confirmed=read(d/'holdout-confirmation.json');models=read(PROJECT/'reports/factorial/preflight/models-and-resume.json')['models'];analysis=read(d/'analysis-manifest.json')
    def get(os,stage,model,mask,repeat=0):
        found=[r for r in matrix if r['os']==os and r['stage']==stage and r['model']==model and int(r['mask'])==mask and int(r['repeat'])==repeat]
        return found[0] if found else None
    def hit(os,stage,model,mask):
        r=get(os,stage,model,mask);return f"{r['hits']}/{r['n']}" if r else '미측정'
    ref='원시 자료와 재현 절차: [research/README.md](research/README.md), [분석 manifest](reports/factorial/analysis/analysis-manifest.json). C=반례, A=적응형, F=강제 수순, T=전술, P=2스레드. P 없는 구성은 1스레드다. 정확 응수 일치율은 Elo가 아니다.'
    note='모든 차이는 퍼센트포인트(pp), 대괄호는 소스 게임 단위 5,000회 짝지은 부트스트랩의 점별 95% 구간이다. 기존 100문제는 선택에 사용한 탐색 자료이며 확정 검증으로 취급하지 않는다. 같은 문제의 반복 실행은 독립 표본으로 세지 않는다.'
    identity=[]
    for m in MODELS:
        r=models[m];identity.append([m,r['training_positions'],r['updates'],r['checkpoint_format'],r['sha256']])
    text=f'''# 신경망 기여도 분석

{ref}

## 모델과 출처

{table(['모델','새 학습 위치','업데이트','PT 포맷','ZERO / 실제 로딩 텐서 SHA-256'],identity)}
모두 ZERO0001 / 입력 1,280 / width64 / residual2 / policy4,672 + WDL3 /
394,179 파라미터다. 모델 구조나 파라미터 수가 증가한 실험이 아니다.
초기 모델은 교사 128개·200스텝이다. 다른 후보는 이 체크포인트에서 추가
학습했으며, 데이터와 업데이트 수·학습 경로가 함께 달라졌다. 이 결과만으로
학습 데이터 수 자체의 인과 효과를 분리할 수 없다. 후보의 고유 학습 위치
수를 초기 128개와 단순 합산하지 않는다.

'''
    for m in MODELS:
        r=models[m];text+=f"### {m}\n\n- 모델: `{r['model_path']}`\n- 체크포인트: `{r['pt_path']}`\n- PT SHA-256: `{r['pt_sha256']}`\n- 시작 PT: `{r['initial_checkpoint_sha256'] or '초기 교사 학습'}`\n\n"
    lineage=read(PROJECT/'reports/factorial/preflight/model-lineage.json')
    text+='### 추가 후보의 보존 bundle 연결\n\n'+table(['후보','bundle ID','commit seq','샤드','처리 원본 행','manifest SHA-256'],[[m,r['bundle_id'],r['commit_seq'],len(r['completed_shards']),r['raw_lines'],r['bundle_manifest_sha256']] for m,r in lineage['candidates'].items()])
    text+='\n각 학습 보고서·manifest의 실제 로컬 경로와 SHA, 원래 cursor와 source identity, PT/미러/ZERO의 동일성 검사는 [model-lineage.json](reports/factorial/preflight/model-lineage.json)에 있다. 실제 학습 실행을 다시 한 것은 아니다.\n\n'
    text+='PT 텐서의 표준 재직렬화 해시, C++가 메모리에서 계산한 실제 로딩 텐서 해시, 파일 해시가 네 모델 모두 일치했다. 각 OS의 절대 로딩 경로는 셀 `.meta.json`의 `loaded_model.path`에 보존했다. C++ 호환성은 전수 실험 중 셀 시작/종료마다 확인했다. 최초 모델의 학습 기록과 10k 계보의 상세 근거는 기존 [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md), 100/1k의 체크포인트 검사는 [preflight](reports/factorial/preflight/models-and-resume.json)에 있다. 원래 학습을 다시 수행한 실험은 아니다.\n\n## 탐색을 고정한 모델 교체\n\n'+note+'\n\n'
    selected=[r for r in model_effects if (r['stage']=='screen' and int(r['mask']) in [0,8,15]) or (r['stage']=='holdout' and int(r['mask']) in [0,15])]
    text+=table(['OS','자료','탐색','후보−초기','교체 효과 pp [95%]','새 정답/손실','Holm p'],[[r['os'],r['stage'],r['factors'],r['model'],interval(r),f"{r['gains']}/{r['losses']}",num(r['p_holm_run'],4)] for r in selected])
    text+='\n## 모델 × 탐색\n\n동일 모델에서의 탐색 효과 차이 `(후보구성−후보기본)−(초기구성−초기기본)`를 모든 공통 구성에서 계산했다. [model-search-interactions.csv](reports/factorial/analysis/model-search-interactions.csv)에 점별·모델 내 동시 구간을 기록했다. 고정된 탐색 구성별로 모델 효과를 구분해야 하며, 각 모델에서 서로 다른 최적 구성을 선택한 점수 차이를 순수 모델 효과로 해석할 수 없다. 별도 학습 seed 반복이나 무작위 데이터 규모 배정이 없어 학습 과정의 불확실성은 이 퍼즐 부트스트랩 구간에 포함되지 않는다.\n'
    write('MODEL_CONTRIBUTION.md',text)

    text=f'''# 탐색 전수 조합 분석

{ref}

## 통제와 실제 구현

네 Boolean 기능 16조합에 P 자원 요인을 교차해 모델당32조합을 측정했다.
네 모델·두 OS의 스크리닝은 총25,600회다. AEGIS는 없으며, SANABI/Laugh
Tale은 외부 구현을 가져온 것이 아닌 이 저장소의 신규 설계다.
정확한 조건·순서·의존성·계측 의미는 [MODULE_AUDIT.md](research/MODULE_AUDIT.md)에 있다.
SharedTree=false 앙상블은 검증 모듈과 양립하지 않아 제외했다.
MateProofNodes=0을 고정했고 기존 기본256은 별도 대조군으로 보존했다.
나머지32조합은 모두 유효하다.

시드1, SIMD, sparse policy, CPuct1.5, FPU0, 64MiB, MaxNodes100000,
검증2048노드/깊이4/후보4다. 각 호스트의 같은 두 CPU ID로 엔진을 제한하고
한 번에 한 셀을 실행했다. P는 두 스레드·예약 배치 스케줄을 함께 바꾸므로
무료 알고리즘 효과로 계산하지 않는다.

## 100ms 스크리닝 전체

![전수 조합](reports/factorial/analysis/figures/factorial-screen.png)

'''
    for os in SYSTEMS:
        text+=f"### {os}\n\n"+table(['mask','구성']+MODELS,[[m,label(m)]+[hit(os,'screen',model,m) for model in MODELS] for m in range(32)])+'\n'
    text+='## 단독 효과와 전체에서 제거한 효과\n\n'+note+'\n\n'
    serial=[r for r in shapley if r['stage']=='screen' and r['design']=='serial4']
    text+=table(['OS','모델','기능','단독 pp','전체−제거 pp','Shapley pp [95%]','CPU Shapley ms','방문 Shapley'],[[r['os'],r['model'],r['factor'],num(r['standalone_pp']),num(r['full_remove_pp']),interval(r,'shapley_pp'),num(r['shapley_cpu_ms']),num(r['shapley_completed_visits'],0)] for r in serial])
    text+='\n![기여도](reports/factorial/analysis/figures/serial-shapley.png)\n\nShapley는 이 지표·문제·시간·요인 집합에 조건부인 기여도다. 알고리즘의 절대적 가치가 아니다. [shapley.csv](reports/factorial/analysis/shapley.csv)에는 5요인, 직렬4요인, 병렬 조건부4요인 및 고정 작업량 결과가 모두 있다. 각 표본에서 기여도 합이 전체−기본과 일치하는지 검사했다. [context-marginals.csv](reports/factorial/analysis/context-marginals.csv)는 모든 배경 구성에서의 추가 효과·새 정답·정답 손실·방문 및 CPU 손익을 제공한다. 단독 효과를 전체 맥락의 기여도와 혼용하지 않는다.\n\n## 선택 고정과 독립 자료\n\n'
    text+=f"Linux 스크리닝만으로 직렬 선택을 `{plan['best_serial']}`, 주 검증 쌍을 `{label(plan['primary_pair_mask'])}`로 고정했다. Windows와 새 자료를 보기 전에 [selection-plan.json](research/selection-plan.json)을 커밋했다.\n\n"
    text+=table(['OS','모델','기본 /200','전체 직렬 /200','고정 선택','선택 /200'],[[os,m,hit(os,'holdout',m,0),hit(os,'holdout',m,15),label(plan['best_serial'][m]),hit(os,'holdout',m,plan['best_serial'][m])] for os in SYSTEMS for m in MODELS])
    text+='\n공식 CC0 자료에서 첫200개 유효한 다른 게임·다른 위치를 동결했다. 기존100문제와 사용 가능한 학습 입력60k 및 초기교사128의 동일 정규화 FEN을 제외했다. 이는 무작위 대국 표본이 아니며, 알려지지 않은 다른 학습 자료와의 중복까지 배제한 것은 아니다. [출처](research/fixtures/holdout.source.json)에는 HTTPS prefix 해시와 원격 전체 프레임 체크섬을 확립하지 못한 범위가 명시돼 있다.\n\n## 시간 반복과 장시간 단계\n\n'
    repeated=[]
    for os in SYSTEMS:
        for m in CORE:
            for mask in sorted({0,8,15,plan['best_serial'][m]}):
                rr=[get(os,'screen',m,mask)]+[get(os,'anchors',m,mask,i) for i in [1,2]];rr=[r for r in rr if r]
                repeated.append([os,m,label(mask),'/'.join(r['hits'] for r in rr),'/'.join(num(r['mean_completed_visits'],0) for r in rr)])
    text+=table(['OS','모델','구성','3회 정답/100','3회 평균 방문'],repeated)
    text+='\n[repeats.json](reports/factorial/analysis/repeats.json)과 [repeat-puzzles.csv](reports/factorial/analysis/repeat-puzzles.csv)는 문제별 정오답·선택 수 변동, 실행 점수 표준편차, 퍼즐별 반복 평균의 게임 단위 구간을 제공한다. 반복 수3은 표본 수100을300으로 늘리지 않는다.\n'
    text+='\n1초는 층화24문제, 5초는 층화16문제, 10초는 사전 규칙으로 고른 정답 획득·손실4사례에 한정했다. 분모가 다른 단계의 점수를 직접 비교하지 않는다. 10초 결과는 진단용으로 일반 정답률로 해석하지 않는다. 모든 단계의 [matrix.csv](reports/factorial/analysis/matrix.csv), 문제별 정오답·선택 수·PV·호출/시간/비용의 [per-puzzle.csv](reports/factorial/analysis/per-puzzle.csv), 각 원시 셀 JSONL.gz를 보존했다.\n'
    text+='\n## 공유된 메이트 한 수 입장 검사의 기여\n\n이 절은 결과 확인 후 추가한 진단이며 주 지표나 구성 선택을 바꾸지 않았다. F/T의 공통 루트 메이트 검사가 깊은 강제 수순 탐색의 효과로 집계되지 않도록 분리했다. `기대 수 M1`은 원래 기대 응수가 즉시 체크메이트인 문제, `공통 M1 증명`은 입장 검사에서 정확히 증명한 횟수다. `동등 M1 허용`은 기대 수도 M1이고 선택 수도 다른 합법 M1일 때만 추가 인정한 민감도 지표다. 다른 장기 승리 수의 정당성은 판정하지 못한다.\n\n'
    diagnostic=[]
    for os in SYSTEMS:
        for stage in ['screen','holdout']:
            for m in CORE:
                for mask in sorted({4,8,15,plan['best_serial'][m]}):
                    r=get(os,stage,m,mask)
                    if r:diagnostic.append([os,stage,m,label(mask),num(float(r['mean_expected_mate1'])*int(r['n']),0),num(float(r['mean_admission_mate1'])*int(r['n']),0),r['hits'],num(float(r['mean_mate_equivalent_hit'])*int(r['n']),0)])
    text+=table(['OS','자료','모델','구성','기대 수 M1','공통 M1 증명','정확 응수','동등 M1 허용'],diagnostic)
    text+='\n[paired-contrasts.csv](reports/factorial/analysis/paired-contrasts.csv)는 새 정답/손실을 기대 수 M1과 기타 문제로 나눈다. 검증기 내부의 재귀 메이트 발견과 공통 루트 스캔은 별개이며, 입장 검사 호출 시간만으로 전체 F/T 인과 비용을 정할 수 없다.\n'
    write('SEARCH_ABLATION.md',text)

    text=f'''# 시너지와 통계적 불확실성

{ref}

## 정의와 확인 기준

`S(A,B)=P(AB)−P(A)−P(B)+P(0)`다. 이는 정확 응수율의 상호작용이며 대국
기력의 시너지를 직접 증명하지 않는다. 모든 비교는 같은 문제·모델·예산에서
짝지었다. 기존100문제는 탐색, 새200문제는 고정된 주 쌍 검증이다.
주 쌍 `{label(plan['primary_pair_mask'])}`는 Linux 스크린의 초기/10k 모델 합산 S가
가장 큰 쌍이다. 모든 주 쌍 OS×모델을 한 가족으로 최대 표준화 오차의
동시95% 부트스트랩 구간, 양 구성 대비 개선의 정확 McNemar 및 Holm 보정을
사용했다. 양 구성보다 유의하게 좋아야 긍정적 시너지를 확인한다.

## 독립 200문제: 주 쌍

'''
    text+=table(['OS','모델','S pp [점별95%]','동시95%','AB−A / AB−B pp','성분 Holm p','확인'],[[r['os'],r['model'],interval(r,'synergy_pp'),f"[{num(r['simultaneous_lo95_pp'])}, {num(r['simultaneous_hi95_pp'])}]",' / '.join(num(t['delta_pp']) for t in r['component_tests']),' / '.join(num(t['p_holm_all_os_models'],4) for t in r['component_tests']),'예' if r['positive_synergy_confirmed'] else '아니오'] for r in confirmed])
    text+='\n[holdout-confirmation.json](reports/factorial/analysis/holdout-confirmation.json)은 새 정답/손실과 가족 크기도 제공한다. 주 쌍 외의 긍정적 상호작용은 독립 자료에서 검증하지 않았으므로 확정하지 않는다.\n\n## 100문제 탐색적 쌍별 상호작용\n\n'
    pairs=[r for r in interactions if r['stage']=='screen' and int(r['order'])==2 and r['model'] in CORE]
    text+=table(['OS','모델','쌍','S pp [점별95%]','같은 모델 모든 차수 동시95%'],[[r['os'],r['model'],r['factors'],interval(r,'interaction_pp'),f"[{num(r['simultaneous_lo95_pp'])}, {num(r['simultaneous_hi95_pp'])}]"] for r in pairs])
    text+='\n## 고차 상호작용\n\n모든 3·4·5요인 Möbius 포함배제 대조를 [interactions.csv](reports/factorial/analysis/interactions.csv)에 저장했다. 아래는 각 차수에서 절댓값이 최대인 탐색 결과이므로 선택 편향이 있다. P가 들어가면 추가 CPU 자원도 포함한다.\n\n'
    higher=[]
    for os in SYSTEMS:
        for m in CORE:
            for order in [3,4,5]:
                rr=[r for r in interactions if r['os']==os and r['model']==m and r['stage']=='screen' and int(r['order'])==order]
                if rr:
                    r=max(rr,key=lambda r:abs(float(r['interaction_pp'])));higher.append([os,m,order,r['factors'],interval(r,'interaction_pp')])
    text+=table(['OS','모델','차수','탐색적 최대 절댓값 항','pp [점별95%]'],higher)
    text+='\n## 품질과 한계\n\n- 게임별 부트스트랩으로 같은 게임의 복수 퍼즐 의존성을 반영했다. 정확 McNemar는 개별 이진 쌍의 독립성을 전제하므로 기존 자료에서는 보조 지표이며 새 자료는 게임당1문제다.\n- 모델 교체 대조는 실행 내 전체 가족 Holm, 기준 대비 탐색 대조는 모델·단계 내 Holm을 적용했다. 이 탐색적 구간들은 전체 모든 지표·예산에 대한 보편적 유의성을 뜻하지 않는다.\n- 주 검증의 동시 가족은 Linux와 Windows의 모든 완료 모델을 함께 포함했다. 반복은 같은 문제를 반복한 것이므로 표본 수를 늘리지 않는다.\n- 100/200개의 고정 prefix 문제는 모집단 무작위 표본이 아니다. 구간은 이 표본에서 소스 게임 재표집이 타당하다는 가정에 의존하며, 다른 퍼즐·대국 전반을 보증하지 않는다.\n- PV는 최종 선택 후 남아 있는 MCTS 트리 경로다. 검증기의 반례/강제 수순 전체 증명 로그로 해석할 수 없다.\n'
    write('SYNERGY_ANALYSIS.md',text)

    cross=rows(d/'cross-platform-cells.csv');numeric=read(d/'numeric-comparison.json');controls=read(d/'cross-platform-controls.json')
    text=f'''# Linux / Windows 차이 진단

{ref}

## 동일성 통제

C++/runner/문제 해시, 모델 파일·로딩 텐서, seed1, 옵션, 스레드 수를 비교했다.
32조합의 OS 간 대조에서 옵션과 문제 순서가 같은지 코드로 검사했다.
서로 다른 호스트·컴파일러다. OS만의 인과 효과로 분리할 수 없다.

'''
    for os,key in [('Linux','linux'),('Windows','windows')]:
        r=controls[key];text+=f"- {os}: `{r['platform']}`, CPU `{r['processor'] or 'environment.json 참조'}`, 논리 CPU {r['logical_cpus']}, 엔진 affinity `{r['engine_cpu_affinity']}`, 실행 파일 SHA `{r['engine_sha256']}`.\n"
    text+='\nLinux의 정확한 CPU명은 실행 폴더 `environment.json`, 컴파일러는 양 OS `compiler.txt`에 보존했다. 별도 artifact UCI 검사 호스트의 CPU명을 본 성능 실험 호스트명으로 대체하지 않는다.\n\n## 1. 같은 입력의 모델 출력\n\n'
    text+=table(['모델','backend','위치','legal 순서 일치','JSON 완전 일치','WDL 최대 절대 오차','legal logit 최대 오차','FP32 최대 ULP'],[[r['model'],r['backend'],r['positions'],r['legal_order_equal'],r['identical_json_positions'],f"{r['wdl_max_abs']:.3g}",f"{r['legal_logit_max_abs']:.3g}",r['max_float32_ulp']] for r in numeric])
    text+='\n100개 입력의 WDL과 모든 합법 수 logit을 SIMD/Scalar 각각 비교했다. 전 policy4,672 전체 원소나 모든 도달 국면의 수치 동등성을 증명한 검사는 아니다.\n\n## 2–4. 고정 작업량\n\n'
    work=[]
    for stage in ['fixed_nodes','fixed_nn','scalar_fixed']:
        for m in MODELS:
            rr=[r for r in cross if r['stage']==stage and r['model']==m]
            if rr:work.append([stage,m,sum(int(r['n']) for r in rr),sum(int(r['same_move']) for r in rr),sum(int(r['same_root_counts']) for r in rr),sum(int(r['same_pv']) for r in rr),max(float(r['max_q_abs']) for r in rr)])
    text+=table(['모드','모델','비교 수','선택 수 일치','루트 방문 일치','PV 일치','Q 최대 절대 차이'],work)
    text+='\n고정 방문512는 검증 추가 작업을 고정하지 않는다. NN512는 root/MCTS/검증을 포함하는 상한이며 실제 평가 수·후보 완료율을 확인했다. Scalar 네 구성은 부동소수점 민감도 대조군이다. 같은 방문 수라도 트리 경로/NN 호출 수가 달라질 수 있어 루트 분포와 PV도 비교했다.\n\n## 5–8. 같은 시간의 처리량과 완료 정도\n\n'
    timing=[]
    for os in SYSTEMS:
        for m in CORE:
            for mask in [0,8,15,plan['best_serial'][m]]:
                r=get(os,'screen',m,mask)
                if r:timing.append([os,m,label(mask),r['hits'],num(r['mean_completed_visits'],0),num(r['mean_nn_total'],0),num(r.get('mean_verification_completed'),2),num(r['mean_search_ms']),num(r['mean_wall_ms']),num(r['mean_mcts_ms']),num(r['mean_verification_ms'])])
    text+=table(['OS','모델','구성','정답/100','방문','NN','검증 완료율','탐색ms','프로토콜ms','MCTS ms','검증ms'],timing)
    text+='\n[cross-platform-puzzles.csv](reports/factorial/analysis/cross-platform-puzzles.csv)에 문제별 선택 수·PV·방문·NN·시간·검증 완료 수·정확 동률 수·최소 양수 선택 간격을 남겼다. PUCT는 엄격한 `>`와 합법 수 생성 순서로 동률을 처리하고, 후보 검증은 1e-5 여유를 둔다. 양 OS 같은 소스다. 작은 추론 오차와 결정 경계의 영향은 관측할 수 있지만, 분기 차이 전부를 부동소수점 한 원인으로 귀속하지 않는다.\n\n[os-search-interactions.csv](reports/factorial/analysis/os-search-interactions.csv)는 같은 문제에서 `(Linux 탐색−Linux 기본)−(Windows 탐색−Windows 기본)`을 계산한다. 하드웨어·컴파일러가 교란된 OS×구성 대조다. 고정 작업량 일치와 고정 시간 불일치가 함께 확인되는 부분은 처리량·완료 여부를 우선 근거로 삼되, CPU/OS/MSVC/GCC 각각의 몫을 분리해 확정하지 않는다. 과거22/45/56과의 차이는 새 계측, 호스트 및 시간 변동도 포함하므로 과거 수치에 소급해 단일 원인을 확정하지 않는다.\n'
    write('CROSS_PLATFORM_ANALYSIS.md',text)

    text=f'''# 계산 비용 및 실행 예산

{ref}

## 예산과 단계 완료

사전 [EXPERIMENT_PLAN.md](research/EXPERIMENT_PLAN.md)에 단계별 예산을
고정했다. OS당 active wall 상한5,400초, CI 별도120분 상한이다.
1차 전수, 후속은 선택된 구성/고정 대조군에 제한했다.

'''
    budget={'screen':1280,'legacy_proof':40,'anchors':160,'one_second':768,'five_seconds':640,'ten_seconds':240,'fixed_nodes':240,'fixed_nn':240,'holdout':480}
    phases=[]
    for run in analysis['runs']:
        root=Path(run['root']);events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
        times=defaultdict(float)
        for e in events:
            if e['event']=='cell_complete':times[e['stage']]+=e['seconds']
        for stage in ['scalar_fixed']+list(budget):
            rr=[r for r in matrix if r['scope']==run['scope'] and r['stage']==stage]
            phases.append([run['manifest']['os'],stage,budget.get(stage,'진단'),len(rr),sum(int(r['n']) for r in rr),num(times[stage],1)])
        text+=f"- {run['manifest']['os']}: pipeline 완료 `{bool(run['completed'])}`, active wall `{num((run['completed'] or run['usage'])['active_seconds'])}`초, 미완료 셀 메타데이터 {len(run['incomplete_cells'])}개.\n"
    text+='\n'+table(['OS','단계','사전 nominal/상한 초','완료 셀','문제 실행','실제 셀 wall초'],phases)
    text+='\n고정 작업량 두 모드의 사전 추정은 합계120–480초이며, 표에서는 각240초로 예산 배분을 표시했다. 셀 wall은 프로세스 시작·모델 확인·warmup·압축 포함, stage 사이 대기·전체 게임 시간은 포함하지 않는다. pipeline active wall이 전체 경과다. 각 실제 호출의 requested budget, search_ns, CPU 및 driver wall은 원시 행에 있다. 10초 문제는 실패/획득 사례를 고른 진단이다.\n\n## 시간·작업량·비용\n\n![CPU 비용](reports/factorial/analysis/figures/accuracy-cpu-cost.png)\n\n'
    cost=[]
    for os in SYSTEMS:
        for stage in ['screen','fixed_nodes','fixed_nn']:
            for m in CORE:
                for mask in sorted({0,8,15,plan['best_serial'][m]}):
                    r=get(os,stage,m,mask)
                    if r:cost.append([os,stage,m,label(mask),f"{r['hits']}/{r['n']}",num(r['mean_cpu_ms']),num(r['mean_wall_ms']),num(r['mean_completed_visits'],0),num(r['mean_node_creations'],0),num(r['mean_edge_expansions'],0),num(r['mean_nn_total'],0),num(r['mean_prune_count'],2)])
    text+=table(['OS','모드','모델','구성','정답','CPU ms','wall ms','완료 방문','새 노드','edge-list 확장','전체 NN','회수'],cost)
    text+='\n전체 평균·중앙값·최대 깊이, 방문 깊이 pooled 통계, 노드/평가/호출/시간·메모리 필드는 [matrix.csv](reports/factorial/analysis/matrix.csv)에 있다. [pareto.csv](reports/factorial/analysis/pareto.csv)는 정확도 대 CPU/wall/NN/방문 비용의 기술적 Pareto 전선을 제공한다. 추정 오차를 반영한 최적 구성 증명은 아니다. 고정 NN에서는 실제 상한 미사용과 후보 미완료도 비용과 함께 해석해야 한다. 고정 방문과 고정 NN 모두 동일 CPU 명령 수를 뜻하지 않는다.\n\n## 계측 비용과 해석\n\n'
    cal=read(PROJECT/'reports/factorial/preflight/calibration.json');by=defaultdict(list)
    for r in cal['observer_effect']:by[r['mask']].append(r['ratio'])
    text+=table(['구성','동일 작업 metrics ON/OFF 시간비 중앙값','최소/최대'],[[label(m),num(statistics.median(v),4),f'{num(min(v),4)} / {num(max(v),4)}'] for m,v in by.items()])
    text+='\n동일512방문·96개 짝 비교다. 병렬 시간 분산과 이상치가 있어 이 비율을 일정한 보정계수로 적용하지 않았다. 계측 ON/OFF와 과거 바이너리의 고정 작업량 결과128개 비교는 일치했다. 시간제어 결과는 계측 ON 연구 바이너리의 결과이며 계측 OFF 배포 엔진의 정답 수로 그대로 주장하지 않는다. 계측은 기본 OFF다.\n\nNN 시간은 MCTS/검증 내부에 있고 병렬 worker 시간은 겹칠 수 있다. F/T 공통 입장 검사 및 NN/물질/정온/반례 exclusive 시간을 분리하되 겹친 시간을 합산하지 않는다. CPU는 OS 계정 해상도 영향이 있으므로 문제당 작은 값보다 셀 합계를 비교한다. RSS는 탐색 후 샘플이며 순간 최대 RSS가 아니다.\n\n## 중단·재개 및 제한\n\n문제마다 JSONL flush+fsync, 완료 시 압축·종료 모델/바이너리 재확인·해시, host/CPU/runner/C++/모델/문제 동일성에 묶인 재개, 동시 writer 거부, 손상된 마지막 행의 보존·복구를 구현하고 검사했다. 중간 손상은 실패 처리한다. 완료된 셀만 분석에 포함하며 `.meta.json`과 압축 해시를 다시 검증했다. 프로세스 강제 종료가 압축 파일과 메타데이터의 여러 파일 갱신 사이에 발생하는 드문 경우는 자동 완결이 보장되지 않는다. 기존 증거를 보존해 별도 실행으로 복구해야 한다.\n'
    text+='\n완료 방문은 신규 계산 위치 수가 아니다. 새 MCTS NN 평가를 뺀 `non_nn_completed_endpoints`에는 캐시된 terminal/depth-limit 값으로 끝난 방문도 있다. `nn_completed_leaf_visits`는 root와 discarded 평가를 제외한다. NN/검색초 및 NN/CPU초는 셀 전체 평가 합계÷시간 합계로 계산해 제공한다. 방문 증가만으로 성능 또는 새 계산량 증가를 주장하지 않는다.\n'
    write('PERFORMANCE_COST.md',text)

    matches=read(d/'matches.json')
    text=f'''# 실전 대국 검증

{ref}

## 조건

Linux 스크리닝에서 선택한 직렬 구성을 각 모델의 기본 MCTS와 비교했다.
초기 모델의 전체 직렬15도 별도 비교했다. 모델/실행 파일 동일, 스레드1,
100ms/move, 동일 호스트 CPU 집합, 고정8오프닝을 색상 교대했다.
게임당 최대200개 신규 ply이며 종료되지 않으면 `*`로 남겼다.
점수·평가에 의한 임의 무승부나 패배 판정은 없다. 같은 규칙으로 청구 가능한
반복/50수 무승부는 python-chess의 claim_draw 규칙을 사용했다.

'''
    text+=table(['OS','모델','구성','W/D/L','미완료','완료 페어','페어 점수','보수적95%','기술적 Elo','Elo 보수적95%'],[[r['os'],r['model'],label(r['mask']),f"{r['wins']}/{r['draws']}/{r['losses']}",r['unfinished'],r['complete_opening_pairs'],num(r['paired_score'],3),'['+', '.join(num(x,3) for x in r['pair_hoeffding_ci95'])+']',str(r['descriptive_elo']),str(r['elo_hoeffding_ci95'])] for r in matches])
    text+='\n[games.csv](reports/factorial/analysis/games.csv)와 [matches.json](reports/factorial/analysis/matches.json)은 게임/페어별 원시 점수, 페어 부트스트랩 및 Hoeffding 구간을 포함한다. 원래 각 실행 폴더 `matches/`에 PGN·전체 UCI trace·모델/바이너리 해시가 있다. 분석기는 PGN과 전체 합법 수, trace, 실제 terminal/claimable outcome 및 후보 색상의 점수가 일치하는지 재생 검사했다.\n\n## 판정\n\n**실제 Elo 개선은 미확정이다.** 사전 기준64개 독립 오프닝 페어에 미달한다. 8개 고정 오프닝의 재표집 구간은 모집단 대국 분포를 보장하지 않으며, 전승 시 무한 Elo 표시는 추정 경계이지 무한한 기력 개선이 아니다. 미완료 게임은 무승부로 합산하지 않고 완료된 양색 페어를 기준으로 계산했다. 모델을 바꾸는 대국은 하지 않았으므로 모델 효과에 대한 대국 Elo 결론도 없다. 전술 최상 구성과 실전 최상 구성이 같다고 확정하지 않았다.\n'
    write('MATCH_VALIDATION.md',text)
    text=f'''# FACTORIAL ABLATION — 최종 답변

{ref}

## 완료 범위와 보존

4모델 × 32조합 × 100ms 스크리닝을 OS별로 실행했다.
모든 완료 셀에서 합계 {sum(int(r['n']) for r in matrix):,}회의 퍼즐 탐색을
분석했다(수치 출력 진단과 대국 제외). 후속1/5/10초, 고정 방문/NN,
새200문제 및 실전 대국은 사전 예산으로 제한했다.
모델·배포·이전 결과를 보존하고 기본 옵션과 후보 승격을 바꾸지 않았다.
코드·테스트·원시 자료·보고서는 operation-overlord에 단계별로 커밋했다.
안전/프로토콜/보존 검사의 상세는
[validation.json](reports/factorial/validation.json)에 있다.

## A. 가장 큰 단독 성능 향상은 무엇인가?

모델·OS에 따라 다르다. 아래는 기존100문제에서 네 직렬 단독 기능 중
최대인 탐색적 효과다. 모델 교체는 같은 탐색에서 별도로 분석했다.

'''
    best_alone=[];best_full=[]
    for os in SYSTEMS:
        for m in MODELS:
            ss=[r for r in serial if r['os']==os and r['model']==m]
            if not ss:continue
            r=max(ss,key=lambda r:float(r['standalone_pp']));best_alone.append([os,m,r['factor'],interval(r,'standalone_pp','standalone_lo95_pp','standalone_hi95_pp')])
            removed=max(ss,key=lambda r:float(r['full_remove_pp']));sh=max(ss,key=lambda r:float(r['shapley_pp']))
            best_full.append([os,m,removed['factor'],interval(removed,'full_remove_pp','remove_lo95_pp','remove_hi95_pp'),sh['factor'],interval(sh,'shapley_pp')])
    text+=table(['OS','모델','최대 단독 기능','기본 대비 pp [95%]'],best_alone)
    text+='\n[MODEL_CONTRIBUTION.md](MODEL_CONTRIBUTION.md)는 모델 고정/교체 대조를, [SEARCH_ABLATION.md](SEARCH_ABLATION.md)는 새 정답·손실과 메이트 한 수 공통 검사를 분리한다. 독립 자료에서 단독 T/A를 다시 측정한 설계는 아니므로 모든 단독 순위를 확정하지 않는다.\n\n## B. 전체 조합에서 가장 큰 기여는 무엇인가?\n\n직렬 전체15에서 하나를 제거한 효과와 모든 배경의 가중 평균 기여도는 다르다. 아래는 각 정의의 최대다.\n\n'
    text+=table(['OS','모델','제거 시 최대 손실 기능','전체−제거 pp [95%]','최대 Shapley 기능','Shapley pp [95%]'],best_full)
    text+='\nP를 포함한 5요인 및 병렬 조건부 기여도는 전체 CSV에 있다. 추가 CPU 자원 효과와 네 기능을 구분했다.\n\n## C. 통계적으로 뒷받침되는 긍정적 시너지가 있는가?\n\n'
    nconfirmed=sum(r['positive_synergy_confirmed'] for r in confirmed)
    text+=f"독립200문제의 고정 주 쌍 `{label(plan['primary_pair_mask'])}`에서 사전 동시구간·성분 대비 Holm 기준을 충족한 모델×OS는 **{nconfirmed}/{len(confirmed)}**개다. "+('해당 범위에서만 정확 응수율의 긍정적 상호작용을 확인했다.' if nconfirmed else '이 설계에서 긍정적 시너지를 확인하지 못했다.')+' 다른 쌍/고차항은 탐색 결과이며 대국 기력 시너지를 입증하지 않는다. [SYNERGY_ANALYSIS.md](SYNERGY_ANALYSIS.md).\n\n## D. 모델 규모가 증가하면 탐색의 가치가 달라지는가?\n\n네 모델은 같은 크기다. 추가 학습 체크포인트에 따라 기여도가 달라지고, 데이터 규모에 대해 단조롭게 증가한다고 결론 내릴 수 없다. 아래는 초기→10k 교체의 구성이 고정된 효과다.\n\n'
    text+=table(['OS','탐색','초기→10k pp [95%]'],[[r['os'],r['factors'],interval(r)] for r in model_effects if r['stage']=='screen' and r['model']=='lichess10000' and int(r['mask']) in [0,8,15]])
    text+='\n모델×탐색의 차이의 차이, 동일 작업량 결과와 다른 후보는 [MODEL_CONTRIBUTION.md](MODEL_CONTRIBUTION.md)를 참조한다. 학습 seed 반복과 데이터 수만의 무작위 개입은 없어 그 인과 효과를 분리하지 못했다.\n\n## E. 동일 시간에서 유리한 구성은 무엇인가?\n\nLinux 스크리닝에서는 F/T 계열을 선택했다. 아래는 선택 이후 새 자료 결과이며, 새 자료를 보고 다시 구성을 고르지 않았다. 보편적 최적 구성이나 모델 승격은 미확정이다.\n\n'
    chosen=[]
    for os in SYSTEMS:
        for m in MODELS:
            mask=plan['best_serial'][m];r=get(os,'holdout',m,mask)
            if not r:continue
            c=next(x for x in contrasts if x['os']==os and x['stage']=='holdout' and x['model']==m and int(x['mask'])==mask)
            chosen.append([os,m,label(mask),hit(os,'holdout',m,0),hit(os,'holdout',m,mask),interval(c),num(r['mean_cpu_ms']),num(r['mean_wall_ms'])])
    text+=table(['OS','모델','고정 후보','기본/200','후보/200','차이 pp [95%]','CPU ms','실제wall ms'],chosen)
    text+='\n## F. 동일 작업량에서 효율적인 구성은 무엇인가?\n\n아래는24문제·32조합 중 정답 수, 이후 CPU 비용으로 정렬한 기술적 후보다. 선택 후 독립 검증한 작업량 최적 구성은 아니다. 512방문은 검증 작업을 고정하지 않고 NN512는 상한을 덜 쓸 수 있으므로 실제 NN과 CPU도 함께 읽어야 한다.\n\n'
    efficient=[]
    for os in SYSTEMS:
        for stage in ['fixed_nodes','fixed_nn']:
            for m in MODELS:
                ss=[r for r in matrix if r['os']==os and r['stage']==stage and r['model']==m]
                if not ss:continue
                r=min(ss,key=lambda r:(-float(r['accuracy']),float(r['mean_cpu_ms']),int(r['mask'])))
                efficient.append([os,stage,m,r['factors'],f"{r['hits']}/{r['n']}",num(r['mean_nn_total']),num(r['mean_completed_visits']),num(r['mean_cpu_ms'])])
    text+=table(['OS','작업 통제','모델','기술적 후보','정답','실제 NN','완료 방문','CPU ms'],efficient)
    text+='\n[PERFORMANCE_COST.md](PERFORMANCE_COST.md)는 Pareto 전선·계측 비용·unused 예산·새 노드/NN과 방문의 차이를 설명한다. 동일 명령 수의 절대 효율 최적은 확립하지 못했다.\n\n## G. Windows에서 낮아지는 원인은 무엇인가?\n\n'
    workrows=[r for r in cross if r['stage']=='fixed_nodes'];total=sum(int(r['n']) for r in workrows);same=sum(int(r['same_move']) for r in workrows)
    text+=f"고정512방문의 선택 수는 전체 {same}/{total}회 일치했다. 수치 출력과 Scalar 대조, 고정NN, 같은 시간의 처리량 및 후보 검증 완료율을 [CROSS_PLATFORM_ANALYSIS.md](CROSS_PLATFORM_ANALYSIS.md)에 나란히 제시했다. 일치하는 고정 작업과 시간에 따른 분기를 구분해야 한다. 두 호스트의 CPU·OS·컴파일러가 함께 달라 순수 OS 원인은 분리되지 않았으며, 과거56 대49–50의 차이를 한 원인으로 확정하지 않는다.\n\n"
    text+='## H. 실제 Elo 개선이 확인됐는가?\n\n**아니다.** 색상 교대·오프닝 페어링·동일 모델/스레드/시간 대국과 PGN 재생 검사는 수행했으나 사전64개 독립 페어 기준에 미달한다. 승무패·미완료·점 추정·보수적 구간은 [MATCH_VALIDATION.md](MATCH_VALIDATION.md)에 있다. 전술 성능과 실전 Elo는 별개이며 어떤 기능도 기본 활성화하거나 후보를 승격하지 않았다.\n'
    write('FACTORIAL_ABLATION.md',text)
    print('rendered six reports and A-H summary')
if __name__=='__main__':main()
