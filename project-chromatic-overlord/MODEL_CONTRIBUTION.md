# 신경망 기여도 분석

원시 자료와 재현 절차: [research/README.md](research/README.md), [분석 manifest](reports/factorial/analysis/analysis-manifest.json). C=반례, A=적응형, F=강제 수순, T=전술, P=2스레드. P 없는 구성은 1스레드다. 정확 응수 일치율은 Elo가 아니다.

## 모델과 출처

| 모델 | 새 학습 위치 | 업데이트 | PT 포맷 | ZERO / 실제 로딩 텐서 SHA-256 |
|---|---|---|---|---|
| zero128 | 128 | 200 | 1 | f6a00902c23e7f32f96b028ac36464cc9012cd8774d5e6b0dee5e1c94e1c03c9 |
| lichess100 | 100 | 12 | 2 | 859a326bf82c02837ef1cbf79538372532412eaee5342db14a539cc1bb3896af |
| lichess1000 | 1000 | 114 | 2 | fcd12bd9438fdadba78e879719714803516a07c4cf1d605583cb2ff013c16bfb |
| lichess10000 | 10000 | 1149 | 2 | 44957f9a35d534883ec92cf015a380dcb20b2fe6096150f687c3a0aa2f75e16a |

모두 ZERO0001 / 입력 1,280 / width64 / residual2 / policy4,672 + WDL3 /
394,179 파라미터다. 모델 구조나 파라미터 수가 증가한 실험이 아니다.
초기 모델은 교사 128개·200스텝이다. 다른 후보는 이 체크포인트에서 추가
학습했으며, 데이터와 업데이트 수·학습 경로가 함께 달라졌다. 이 결과만으로
학습 데이터 수 자체의 인과 효과를 분리할 수 없다. 후보의 고유 학습 위치
수를 초기 128개와 단순 합산하지 않는다.

### zero128

- 모델: `/workspace/Chromatic/chromatic-shard-trainer/champion/default.zero`
- 체크포인트: `/workspace/Chromatic/chromatic-shard-trainer/champion/model.pt`
- PT SHA-256: `99151f0c57212edd6786e2bbbafc1717bb15bee9a24341e69c5606c478bb4b84`
- 시작 PT: `초기 교사 학습`

### lichess100

- 모델: `/workspace/Chromatic/chromatic-shard-trainer/candidates/positions-100.zero`
- 체크포인트: `/workspace/Chromatic/chromatic-shard-trainer/candidates/positions-100.pt`
- PT SHA-256: `2385321f8917d91e32c836f2d17641f3ddfeeb825befff63bbffabd039c6ffaf`
- 시작 PT: `99151f0c57212edd6786e2bbbafc1717bb15bee9a24341e69c5606c478bb4b84`

### lichess1000

- 모델: `/workspace/Chromatic/chromatic-shard-trainer/candidates/positions-1000.zero`
- 체크포인트: `/workspace/Chromatic/chromatic-shard-trainer/candidates/positions-1000.pt`
- PT SHA-256: `6f5af8b9d0d95b442561bab53de04ae4120922e3b8e5ddd3b7289105c0696867`
- 시작 PT: `99151f0c57212edd6786e2bbbafc1717bb15bee9a24341e69c5606c478bb4b84`

### lichess10000

- 모델: `/workspace/Chromatic/chromatic-shard-trainer/candidates/positions-10000.zero`
- 체크포인트: `/workspace/Chromatic/chromatic-shard-trainer/candidates/positions-10000.pt`
- PT SHA-256: `427da59067d8acf1a799ad57ff8dbbceb327d070aeed4fb9fc12bb3911453143`
- 시작 PT: `99151f0c57212edd6786e2bbbafc1717bb15bee9a24341e69c5606c478bb4b84`

### 추가 후보의 보존 bundle 연결

| 후보 | bundle ID | commit seq | 샤드 | 처리 원본 행 | manifest SHA-256 |
|---|---|---|---|---|---|
| lichess100 | e4354ee1423f4060ab079a815ee5a6dc | 9 | 4 | 256 | 93048fb4b3bd83ecd88ca00b78208f7800cf5226dbd2b934a60c41eebfd32938 |
| lichess1000 | 6bbfd686b2254f12a54b75dc242e1a8a | 12 | 4 | 1536 | d32d6656118c17ce214b8b08fda2929385e2d9d36156e6d1444c293eceb2c2b1 |
| lichess10000 | a181d36c81b94b22aa786094e626b6b8 | 56 | 10 | 25728 | 3382c0974443ff960c5f14dd04804e42e10e7b0014726d929a9b81d3c2a8db6b |

각 학습 보고서·manifest의 실제 로컬 경로와 SHA, 원래 cursor와 source identity, PT/미러/ZERO의 동일성 검사는 [model-lineage.json](reports/factorial/preflight/model-lineage.json)에 있다. 실제 학습 실행을 다시 한 것은 아니다.

PT 텐서의 표준 재직렬화 해시, C++가 메모리에서 계산한 실제 로딩 텐서 해시, 파일 해시가 네 모델 모두 일치했다. 각 OS의 절대 로딩 경로는 셀 `.meta.json`의 `loaded_model.path`에 보존했다. C++ 호환성은 전수 실험 중 셀 시작/종료마다 확인했다. 최초 모델의 학습 기록과 10k 계보의 상세 근거는 기존 [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md), 100/1k의 체크포인트 검사는 [preflight](reports/factorial/preflight/models-and-resume.json)에 있다. 원래 학습을 다시 수행한 실험은 아니다.

## 탐색을 고정한 모델 교체

모든 차이는 퍼센트포인트(pp), 대괄호는 소스 게임 단위 5,000회 짝지은 부트스트랩의 점별 95% 구간이다. 기존 100문제는 선택에 사용한 탐색 자료이며 확정 검증으로 취급하지 않는다. 같은 문제의 반복 실행은 독립 표본으로 세지 않는다.

| OS | 자료 | 탐색 | 후보−초기 | 교체 효과 pp [95%] | 새 정답/손실 | Holm p |
|---|---|---|---|---|---|---|
| Linux | holdout | base | lichess100 | 6.0 [0.5, 11.5] | 21/9 | 1 |
| Linux | holdout | base | lichess1000 | 12.5 [6.5, 18.5] | 34/9 | 0.056 |
| Linux | holdout | base | lichess10000 | 21.5 [15.5, 28.0] | 48/5 | 2.34e-07 |
| Linux | holdout | CAFT | lichess100 | -2.5 [-8.0, 3.0] | 14/19 | 1 |
| Linux | holdout | CAFT | lichess1000 | 3.0 [-2.5, 9.0] | 21/15 | 1 |
| Linux | holdout | CAFT | lichess10000 | 4.0 [-2.5, 10.5] | 28/20 | 1 |
| Linux | screen | base | lichess100 | 5.0 [-3.0, 13.0] | 11/6 | 1 |
| Linux | screen | base | lichess1000 | 9.0 [0.0, 18.0] | 15/6 | 1 |
| Linux | screen | base | lichess10000 | 13.0 [4.0, 22.0] | 18/5 | 1 |
| Linux | screen | T | lichess100 | 3.0 [-3.0, 9.0] | 6/3 | 1 |
| Linux | screen | T | lichess1000 | 6.0 [-1.0, 13.0] | 10/4 | 1 |
| Linux | screen | T | lichess10000 | 5.0 [-3.0, 13.0] | 12/7 | 1 |
| Linux | screen | CAFT | lichess100 | -8.0 [-17.0, 1.0] | 6/14 | 1 |
| Linux | screen | CAFT | lichess1000 | -8.0 [-15.0, -1.0] | 3/11 | 1 |
| Linux | screen | CAFT | lichess10000 | -5.0 [-14.0, 3.0] | 7/12 | 1 |
| Windows | holdout | base | lichess100 | 5.5 [0.5, 10.5] | 20/9 | 1 |
| Windows | holdout | base | lichess1000 | 11.0 [5.0, 17.0] | 32/10 | 0.309 |
| Windows | holdout | base | lichess10000 | 21.0 [14.5, 27.5] | 48/6 | 1.08e-06 |
| Windows | holdout | CAFT | lichess100 | -3.0 [-8.5, 2.5] | 13/19 | 1 |
| Windows | holdout | CAFT | lichess1000 | 2.5 [-3.0, 8.0] | 20/15 | 1 |
| Windows | holdout | CAFT | lichess10000 | 5.0 [-2.0, 12.0] | 29/19 | 1 |
| Windows | screen | base | lichess100 | 4.0 [-4.0, 12.0] | 10/6 | 1 |
| Windows | screen | base | lichess1000 | 8.0 [0.0, 17.0] | 14/6 | 1 |
| Windows | screen | base | lichess10000 | 12.0 [3.0, 21.0] | 17/5 | 1 |
| Windows | screen | T | lichess100 | 2.0 [-3.0, 8.0] | 5/3 | 1 |
| Windows | screen | T | lichess1000 | 6.0 [-1.0, 13.0] | 10/4 | 1 |
| Windows | screen | T | lichess10000 | 3.0 [-5.0, 11.0] | 10/7 | 1 |
| Windows | screen | CAFT | lichess100 | -2.0 [-10.0, 5.0] | 6/8 | 1 |
| Windows | screen | CAFT | lichess1000 | -4.0 [-10.0, 2.0] | 3/7 | 1 |
| Windows | screen | CAFT | lichess10000 | -1.0 [-8.0, 6.0] | 6/7 | 1 |

## 모델 × 탐색

동일 모델에서의 탐색 효과 차이 `(후보구성−후보기본)−(초기구성−초기기본)`를 모든 공통 구성에서 계산했다. [model-search-interactions.csv](reports/factorial/analysis/model-search-interactions.csv)에 점별·모델 내 동시 구간을 기록했다. 고정된 탐색 구성별로 모델 효과를 구분해야 하며, 각 모델에서 서로 다른 최적 구성을 선택한 점수 차이를 순수 모델 효과로 해석할 수 없다. 별도 학습 seed 반복이나 무작위 데이터 규모 배정이 없어 학습 과정의 불확실성은 이 퍼즐 부트스트랩 구간에 포함되지 않는다.
