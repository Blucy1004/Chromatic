# PROJECT CHROMATIC 1.0 — OPERATION OVERDRIVE

PROJECT ZERO v0.1에서 이어진 독립 C++20 PUCT-MCTS + PyTorch Policy/WDL 엔진입니다. 이번 릴리스는 성능·정확성·학습 신뢰성 개선을 검증한 엔지니어링 릴리스입니다. 아직 통계적으로 확정된 Elo 향상이나 무영과의 대결 성능을 주장하지 않습니다.

## 실행

`dist/chromatic.exe`를 UCI 체스 GUI에 등록하세요. 같은 폴더의 `default.zero`를 자동으로 읽습니다. Python·PyTorch·외부 체스 엔진을 호출하지 않으며 Windows의 표준 MSVC 런타임을 사용합니다. 배포 기본 모델은 검증된 v0.1 부트스트랩 가중치를 유지합니다. 새 학습 후보는 대결 증거가 충분할 때만 승격됩니다.

기본값은 FP32 SIMD, 합법 정책 출력 계산, cpuct=1.5, FPU reduction=0, 최대 보존 트리 메모리 64MiB, 1수 메이트 보호입니다. 더 깊은 진단용 메이트 증명은 `MateProofPlies=5`, `MateProofNodes=12000`으로 선택할 수 있습니다. 깊은 옵션은 더 많은 계산을 소비하므로 기본값으로 선택하지 않았습니다.

```text
uci
isready
position startpos moves e2e4 e7e5
go movetime 100
stop
quit
```

## 빌드와 검증

```powershell
.\build.cmd
.\validate.ps1
.\asan.cmd
.\build.cmd mutation
.\run.ps1 --help
```

`validate.ps1`는 perft/핵심 C++ 테스트와 통합·학습 신뢰성·대결/롤백·입력 안전성 테스트를 실행합니다. `asan.cmd`는 별도 빌드에서 AddressSanitizer 검사를 수행합니다. 변이 테스트의 자동 판정은 다음과 같습니다.

```powershell
.\.venv\Scripts\python.exe tools\mutation.py --executable build\sign_mutant.exe --output reports\sign-mutation.json
```

소스 빌드 결과의 이름은 기존 도구와의 호환을 위해 `build/project_zero.exe`이며, UCI ID는 `CHROMATIC 1.0`입니다. 배포 파일 `dist/chromatic.exe`는 이 빌드와 동일한 바이너리입니다. 기본 빌드는 지역화된 컴파일러의 오래된 헤더 의존성 문제를 피하도록 전체 재빌드를 수행합니다.

이번 컴퓨터에서는 작업 전용 Python을 `run.ps1`가 자동으로 찾습니다. 다른 컴퓨터에서 학습·테스트하려면 Python 3.12 이상으로 전용 환경을 만드세요.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

기준선과 제한된 전체 실험을 새 폴더에서 재현하는 도구도 있습니다. 아래 전체 실험은 4판 자가대국/100스텝과 작은 대결들로 제한됩니다.

```powershell
.\.venv\Scripts\python.exe tools\baseline_reconstruct.py --output reports\baseline-rerun
.\.venv\Scripts\python.exe tools\final_experiments.py --output reports\bounded-rerun
.\.venv\Scripts\python.exe tools\legacy_compat.py
```

## 기준선 보존과 호환성

`baseline/project-zero-v0.1.zip`은 원본 배포 ZIP, `baseline/project_zero.exe`와 `baseline/default.zero`는 기준 실행파일·모델입니다. `baseline/source`는 읽기와 재현을 위한 원본 소스입니다. `baseline/instrumented`는 시간 계측만 추가한 기준 코어이며 같은 계산과 출력 체크섬을 유지합니다. 기존 PROJECT ZERO 폴더와 무영 저장소는 수정하지 않았습니다.

* 신경망 파일은 기존 `ZERO0001`, 20×64 입력, 4,672개 정책 인덱스를 그대로 사용합니다.
* 체크포인트 reader는 v1과 v2를 읽습니다. 새 저장은 v2입니다.
* `--init`는 v0.1 모델을 새 학습으로 가져옵니다. 원래 옵티마이저와 샘플러를 그대로 이어갈 때는 명시적으로 `train --legacy --resume ...`를 사용하세요. 원본 소스와의 201스텝 결과 동등성을 검사했습니다.
* 자가대국 record v2는 압축된 입력, 합법 action 목록, 선택한 action, seed, 검색 예산과 실제 종료 이유를 추가합니다. v1 데이터도 계속 읽습니다. v2 데이터는 v0.1 학습기로 직접 읽을 수 없습니다.
* SIMD는 같은 설정/장치에서 결정적입니다. 이전 스칼라 산술과 정확한 탐색 경로를 대조하려면 `InferenceBackend=Scalar`, `MateProofNodes=0`을 사용하세요.

## 자가대국과 학습

```powershell
.\run.ps1 selfplay --executable build\project_zero.exe --model runs\bootstrap\model.zero --output runs\games --games 4 --simulations 64 --max-plies 512 --opening-plies 6
.\run.ps1 train --data 'runs/games/game_*.jsonl' --init runs\bootstrap\model.pt --output runs\candidate.pt --steps 200 --capacity 10000 --validation-fraction 0.15
.\run.ps1 train --data 'runs/games/game_*.jsonl' --resume runs\candidate.pt --output runs\candidate.pt --steps 400 --capacity 10000 --validation-fraction 0.15
```

`--steps`는 총 목표 스텝입니다. Resume는 데이터 해시와 순서, 배치·옵티마이저·학습률·분리 설정·PyTorch 버전이 같아야 합니다. 체크포인트는 가중치, 옵티마이저 모멘트, 스케줄 위치, 모든 RNG, 샘플러와 최근/과거 샘플 수를 보존합니다. 손상된 최신 파일은 `.prev`에서 복구합니다.

Replay는 기본 10,000개 기록으로 제한합니다. 최신 입력 디렉터리와 이전 디렉터리에서 최근/과거 표본을 섞고 실제 선택·학습 비율을 기록합니다. 중복 검사는 용량의 8배인 한정된 창에서 수행하며 전역 중복 제거라고 주장하지 않습니다. 같은 대국이나 같은 입력 위치는 학습/검증 양쪽으로 나누지 않습니다. 분리할 독립 그룹이 부족하면 검증 지표가 없다고 명시합니다.

미완료 대국은 무승부로 학습하지 않습니다. 결과를 모르는 방문 분포는 `pending/`에 보존하고, 완료된 최상위 JSONL만 학습에 사용합니다. 수 제한을 늘려 계속할 때는 다음처럼 명시하세요.

```powershell
.\run.ps1 selfplay --executable build\project_zero.exe --model runs\bootstrap\model.zero --output runs\games --games 4 --simulations 64 --max-plies 1024 --opening-plies 6 --resume-truncated
```

자동 세대 루프는 교사 없이 실행합니다.

```powershell
.\run.ps1 loop --executable build\project_zero.exe --initial runs\bootstrap\model.pt --output runs\new-cycle --iterations 2 --games 8 --simulations 128 --steps 500 --pairs 16 --max-plies 512 --arena-milliseconds 20 --opening-suite benchmarks\openings.json
```

같은 명령을 다시 실행하면 완료 세대와 체크섬이 맞는 대국을 재사용하고 학습을 재개합니다. `--iterations`는 누적 목표입니다. 다른 바이너리나 설정에는 새 출력 폴더를 사용하고 한 출력 폴더를 동시에 여러 프로세스에서 쓰지 마세요. 장시간 학습이나 대규모 대결은 별도 예산을 정한 뒤 실행하세요.

## 대결과 롤백

```powershell
.\run.ps1 arena --executable build\project_zero.exe --incumbent-executable baseline\project_zero.exe --candidate baseline\default.zero --incumbent baseline\default.zero --output runs\comparison.json --pairs 8 --milliseconds 20 --max-plies 512 --opening-suite benchmarks\openings.json
.\run.ps1 rollback --output runs\new-cycle --steps 1
```

대결은 같은 CPU 코어, 같은 한 수 시간/노드 예산, 같은 오프닝의 색 교환을 지원하고 PGN·WDL·불완료·오류·보수적 신뢰구간을 저장합니다. 미완료 대국이 있으면 Elo를 계산하지 않습니다. 증거 부족과 열세가 입증된 경우를 구별합니다. 승격은 오프닝 쌍 단위 신뢰하한이 기준을 통과할 때만 허용하며, rollback은 이전 모델 해시를 확인한 뒤 복원합니다.

`--sprt`는 비긴 오프닝 쌍을 제외한 쌍 승패의 정확한 Bernoulli sign SPRT입니다. 표준 Elo 가설 SPRT로 부르지 않습니다. 임의 중지 시 승격 신뢰구간에는 중지 시점 전체를 고려한 보수적 보정을 적용합니다. 작은 릴리스 대결에서는 사용하지 않았습니다.

## 보고서

* [성능 보고서](PERFORMANCE.md): 고정 코어 기준 측정, 정책 헤드/SIMD 분리 비교, 메모리, 두 모델 크기.
* [정확성 감사](CORRECTNESS_AUDIT.md): 발견한 결함, 부호 변이, perft, 모델/복구/메모리 검증.
* [실력 검증](STRENGTH_REPORT.md): 전술 문제, 동일 시간 대결, 자가학습 결과와 판단 한계.
* [구조와 형식](ARCHITECTURE.md): 버전, 인코딩, PUCT, 트리 수명, 학습과 승격.
* [남은 작업](ROADMAP.md): 아직 구현·검증하지 않은 영역.

테스트와 수치는 모두 실제 실행한 JSON·PGN으로 연결됩니다. 네트워크 크기, 탐색량, 낮은 학습 loss를 Elo 향상으로 바꾸어 주장하지 않습니다.
