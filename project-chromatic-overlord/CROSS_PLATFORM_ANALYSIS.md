# Linux / Windows 차이 진단

원시 자료와 재현 절차: [research/README.md](research/README.md), [분석 manifest](reports/factorial/analysis/analysis-manifest.json). C=반례, A=적응형, F=강제 수순, T=전술, P=2스레드. P 없는 구성은 1스레드다. 정확 응수 일치율은 Elo가 아니다.

## 동일성 통제

C++/runner/문제 해시, 모델 파일·로딩 텐서, seed1, 옵션, 스레드 수를 비교했다.
32조합의 OS 간 대조에서 옵션과 문제 순서가 같은지 코드로 검사했다.
서로 다른 호스트·컴파일러다. OS만의 인과 효과로 분리할 수 없다.

- Linux: `Linux-6.18.55-x86_64-with-glibc2.41`, CPU `environment.json 참조`, 논리 CPU 5, 엔진 affinity `[0, 1]`, 실행 파일 SHA `b97bcfc96400e6b7973382bf30234654c82cb97181eb68b86ba79ebe1cbf5428`.
- Windows: `Windows-2022Server-10.0.20348-SP0`, CPU `AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD`, 논리 CPU 4, 엔진 affinity `[0, 1]`, 실행 파일 SHA `8eb4dc51c223a74ef18f310950292ff9cfb37341cb1593f9662c50b0f06d3cc8`.

Linux의 정확한 CPU명은 실행 폴더 `environment.json`, 컴파일러는 양 OS `compiler.txt`에 보존했다. 별도 artifact UCI 검사 호스트의 CPU명을 본 성능 실험 호스트명으로 대체하지 않는다.

## 1. 같은 입력의 모델 출력

| 모델 | backend | 위치 | legal 순서 일치 | JSON 완전 일치 | WDL 최대 절대 오차 | legal logit 최대 오차 | FP32 최대 ULP |
|---|---|---|---|---|---|---|---|
| zero128 | SIMD | 100 | 100 | 100 | 0 | 0 | 0 |
| zero128 | Scalar | 100 | 100 | 100 | 0 | 0 | 0 |
| lichess100 | SIMD | 100 | 100 | 100 | 0 | 0 | 0 |
| lichess100 | Scalar | 100 | 100 | 99 | 3.7e-09 | 0 | 2 |
| lichess1000 | SIMD | 100 | 100 | 100 | 0 | 0 | 0 |
| lichess1000 | Scalar | 100 | 100 | 100 | 0 | 0 | 0 |
| lichess10000 | SIMD | 100 | 100 | 100 | 0 | 0 | 0 |
| lichess10000 | Scalar | 100 | 100 | 99 | 1.8e-09 | 0 | 2 |

100개 입력의 WDL과 모든 합법 수 logit을 SIMD/Scalar 각각 비교했다. 전 policy4,672 전체 원소나 모든 도달 국면의 수치 동등성을 증명한 검사는 아니다.

## 2–4. 고정 작업량

| 모드 | 모델 | 비교 수 | 선택 수 일치 | 루트 방문 일치 | PV 일치 | Q 최대 절대 차이 |
|---|---|---|---|---|---|---|
| fixed_nodes | zero128 | 768 | 768 | 768 | 768 | 5.9604000068880225e-08 |
| fixed_nodes | lichess100 | 768 | 768 | 768 | 768 | 0.0 |
| fixed_nodes | lichess1000 | 768 | 768 | 768 | 768 | 7.450599998071183e-09 |
| fixed_nodes | lichess10000 | 768 | 768 | 768 | 768 | 1.4901000017220056e-08 |
| fixed_nn | zero128 | 768 | 768 | 768 | 768 | 5.9604000068880225e-08 |
| fixed_nn | lichess100 | 768 | 768 | 768 | 768 | 0.0 |
| fixed_nn | lichess1000 | 768 | 768 | 768 | 768 | 3.725299999035592e-09 |
| fixed_nn | lichess10000 | 768 | 768 | 768 | 768 | 1.4901000017220056e-08 |
| scalar_fixed | zero128 | 96 | 96 | 96 | 96 | 0.0 |
| scalar_fixed | lichess100 | 96 | 96 | 96 | 96 | 0.0 |
| scalar_fixed | lichess1000 | 96 | 96 | 96 | 96 | 0.0 |
| scalar_fixed | lichess10000 | 96 | 96 | 96 | 96 | 7.45060000154063e-09 |

고정 방문512는 검증 추가 작업을 고정하지 않는다. NN512는 root/MCTS/검증을 포함하는 상한이며 실제 평가 수·후보 완료율을 확인했다. Scalar 네 구성은 부동소수점 민감도 대조군이다. 같은 방문 수라도 트리 경로/NN 호출 수가 달라질 수 있어 루트 분포와 PV도 비교했다.

## 5–8. 같은 시간의 처리량과 완료 정도

| OS | 모델 | 구성 | 정답/100 | 방문 | NN | 검증 완료율 | 탐색ms | 프로토콜ms | MCTS ms | 검증ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Linux | zero128 | base | 22 | 65436 | 3948 | 미측정 | 100.2 | 100.6 | 100.0 | 0.0 |
| Linux | zero128 | T | 45 | 47360 | 3063 | 1.00 | 75.7 | 76.1 | 75.0 | 0.7 |
| Linux | zero128 | CAFT | 56 | 47710 | 3488 | 0.78 | 84.9 | 85.2 | 73.2 | 11.6 |
| Linux | zero128 | AFT | 63 | 46908 | 2990 | 1.00 | 74.5 | 74.8 | 73.2 | 1.2 |
| Linux | lichess10000 | base | 35 | 79200 | 3760 | 미측정 | 100.2 | 100.6 | 100.0 | 0.0 |
| Linux | lichess10000 | T | 50 | 56617 | 2901 | 1.00 | 75.9 | 76.3 | 75.0 | 0.8 |
| Linux | lichess10000 | CAFT | 51 | 53586 | 3389 | 0.65 | 86.1 | 86.4 | 75.0 | 10.9 |
| Linux | lichess10000 | FT | 62 | 53709 | 2886 | 1.00 | 76.6 | 76.9 | 75.0 | 1.5 |
| Windows | zero128 | base | 22 | 41407 | 2944 | 미측정 | 100.1 | 101.0 | 100.0 | 0.0 |
| Windows | zero128 | T | 45 | 30098 | 2271 | 1.00 | 76.0 | 76.6 | 75.0 | 0.9 |
| Windows | zero128 | CAFT | 48 | 30506 | 2686 | 0.67 | 88.0 | 88.6 | 73.2 | 14.7 |
| Windows | zero128 | AFT | 63 | 30437 | 2233 | 1.00 | 74.9 | 75.5 | 73.2 | 1.6 |
| Windows | lichess10000 | base | 34 | 50674 | 2824 | 미측정 | 100.1 | 100.7 | 100.0 | 0.0 |
| Windows | lichess10000 | T | 48 | 36353 | 2163 | 1.00 | 76.3 | 76.9 | 75.0 | 1.2 |
| Windows | lichess10000 | CAFT | 47 | 34564 | 2619 | 0.58 | 88.5 | 89.1 | 75.0 | 13.4 |
| Windows | lichess10000 | FT | 62 | 36234 | 2198 | 1.00 | 77.1 | 77.7 | 75.0 | 2.0 |

[cross-platform-puzzles.csv](reports/factorial/analysis/cross-platform-puzzles.csv)에 문제별 선택 수·PV·방문·NN·시간·검증 완료 수·정확 동률 수·최소 양수 선택 간격을 남겼다. PUCT는 엄격한 `>`와 합법 수 생성 순서로 동률을 처리하고, 후보 검증은 1e-5 여유를 둔다. 양 OS 같은 소스다. 작은 추론 오차와 결정 경계의 영향은 관측할 수 있지만, 분기 차이 전부를 부동소수점 한 원인으로 귀속하지 않는다.

[os-search-interactions.csv](reports/factorial/analysis/os-search-interactions.csv)는 같은 문제에서 `(Linux 탐색−Linux 기본)−(Windows 탐색−Windows 기본)`을 계산한다. 하드웨어·컴파일러가 교란된 OS×구성 대조다. 고정 작업량 일치와 고정 시간 불일치가 함께 확인되는 부분은 처리량·완료 여부를 우선 근거로 삼되, CPU/OS/MSVC/GCC 각각의 몫을 분리해 확정하지 않는다. 과거22/45/56과의 차이는 새 계측, 호스트 및 시간 변동도 포함하므로 과거 수치에 소급해 단일 원인을 확정하지 않는다.

## 관측된 원인 범위

SIMD는400개 모델×입력의 WDL/합법 logits가 모두 일치했고, 고정512방문과 NN512 상한 각각3,072회의 선택 수도 모두 일치했다. 같은 시간에서는 처리한 NN 수와 후보 완료율이 달랐다. 이는 이 검사 범위에서 시간에 따른 처리량·배분 차이를 우선 설명으로 지지한다. CPU 주파수·코어/SMT 배치·컴파일러·OS clock 비용을 각각 분리한 실험은 아니다. 숫자 affinity ID가 같다고 같은 물리 코어 구성을 뜻하지 않는다. Linux 계측 ON/OFF 비용은 측정했으나 Windows의 동일 비용 대조는 별도로 측정하지 않았다.

## 10초 진단 표본의 제외

10초 사례는 각 모델/OS의 로컬 스크린 획득·손실로 골랐다. Windows는 같은 구성 계획을 사용했지만 이 사례 ID는 일부 달랐다. 분석기가 이를 발견해 중단한 첫 로그를 [analysis-failures](reports/factorial/analysis-failures/)에 보존했다. [excluded-cross-platform-cells.json](reports/factorial/analysis/excluded-cross-platform-cells.json)에 다른 ID와 제외 이유를 명시했으며, 맞지 않는 표본을 OS 차이로 계산하지 않았다. 모델 간10초 대조도 전체 제외했다. 같은 모델/OS 내10초 구성 대조와 원시 결과는 남아 있다. 주 스크린·고정 작업량·독립200문제의 표본 짝은 계속 엄격히 검사했다.
