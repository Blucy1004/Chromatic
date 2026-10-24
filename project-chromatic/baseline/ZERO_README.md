# PROJECT ZERO 0.1

독립 C++20 PUCT-MCTS 체스 엔진과 PyTorch Policy/WDL 학습 파이프라인입니다. 소규모 교사 부트스트랩 이후에는 외부 엔진 없이 자가대국 결과로 학습합니다. 현재 릴리스는 전체 흐름과 정확성 검증을 완료한 초기 구현이며, 대규모 학습이나 Elo 강도 검증은 수행하지 않았습니다.

## 바로 실행

`dist/project_zero.exe`를 UCI 지원 체스 GUI에 엔진으로 등록하세요. 같은 폴더의 `default.zero`를 자동으로 읽습니다. 이 모델은 128개 교사 위치로 200스텝 학습한 초기 모델입니다. 학습 후보를 자동 승격할 근거는 아직 확보되지 않았습니다.

엔진 실행에는 Python/PyTorch/Stockfish가 필요하지 않습니다. Windows x64에서 빌드했습니다. 다른 운영체제에서는 CMake로 소스를 빌드할 수 있지만 이번 검증은 Windows에서만 수행했습니다.

소스 빌드와 검증:

```powershell
.\build.cmd
.\validate.ps1
```

이 컴퓨터에서는 `run.ps1`와 `validate.ps1`가 이번 작업 전용 Python 환경을 찾습니다. 다른 곳으로 복사한 경우 Python 3.12 이상의 전용 환경을 만드세요.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\run.ps1 --help
```

`requirements.lock.txt`는 검증에 사용한 정확한 버전입니다. 일반적인 새 설치에는 `requirements.txt`도 사용할 수 있습니다. CUDA 학습에는 해당 장치와 맞는 CUDA용 PyTorch가 필요하며, 이번에 설치·검증한 환경은 CPU용입니다.

## 단계별 구현 결과

| 단계 | 구현 | 확인 |
|---|---|---|
| 1. 규칙 | FEN, 합법 수, 캐슬링, 앙파상, 승격, 체크·메이트·스테일메이트 | 5개 perft 위치, 독립 라이브러리 1,008개 위치 대조 |
| 2. 탐색/UCI | PUCT, legal policy mask, WDL 역전파, root noise, 온도 샘플링, 비동기 UCI | 메이트, 방문 수 보존, 시간 제한, stop, 탐색 중 isready |
| 3. 모델 | PyTorch residual MLP Policy/WDL, 원자적 FP32 export, 독립 C++ 추론 | 초기 모델 52개 위치 및 학습 모델 100개 위치 일치 |
| 4. 학습/복구 | AdamW, 방문 분포+WDL 학습, 옵티마이저·RNG·샘플러 복구 | 연속/중단 학습 동일, 손상 파일에서 .prev 복구 |
| 5. 자가학습 | 소량 교사, 자가대국 shard, 최근 세대 replay, 후보 대결, 승격 판정 | 실제 시작 위치에서 한 세대 실행, 세대 재시작 테스트 |

## 교사 부트스트랩

교사는 별도 실행파일로만 호출합니다. 교사 소스나 탐색 코드를 C++ 엔진에 포함하지 않습니다. `bootstrap`은 한 번에 최대 10,000개 위치로 제한하며 기본값은 128개입니다. 교사 실행파일의 SHA-256과 엔진 ID, 노드 예산, WDL 라벨 출처를 기록합니다.

```powershell
.\run.ps1 bootstrap --teacher C:\path\to\stockfish.exe --output data\teacher.jsonl --positions 128 --nodes 2000 --multipv 4 --seed 2026
.\run.ps1 train --data data\teacher.jsonl --output models\bootstrap.pt --steps 200 --batch-size 32 --save-every 50 --seed 2026
```

교사가 WDL을 제공하면 그 분포를 사용하고, 없으면 `cp-logistic-400`이라는 명시적 근사 라벨을 사용합니다. 이 근사는 부트스트랩에만 적용됩니다. 재현용 교사는 [Stockfish 공식 배포](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19)에서 받은 Stockfish 19였습니다.

## 자가대국 학습

```powershell
.\run.ps1 loop --executable build\project_zero.exe --initial runs\bootstrap\model.pt --output runs\next --iterations 3 --games 32 --simulations 256 --steps 1000 --batch-size 64 --pairs 64 --max-plies 512 --seed 2026
```

한 세대는 현재 챔피언으로 대국을 만들고, 최근 4개 세대의 **자가대국 데이터만** 학습한 뒤, 후보와 챔피언을 같은 오프닝에서 색을 바꾸어 대결시킵니다. 교사 데이터는 이 replay에 섞지 않습니다. 세대별 `source.zero`, `candidate.pt`, `candidate.zero`, `arena.json`과 전체 `state.json`을 남깁니다.

같은 명령을 다시 실행하면 완료한 대국 shard를 체크섬으로 확인하여 재사용하고, 학습 체크포인트에서 재개하며, 완료한 세대는 건너뜁니다. `--iterations`는 추가 개수가 아닌 **목표 누적 세대 수**입니다. 다른 설정·다른 엔진 바이너리를 사용하려면 새 출력 폴더를 사용하세요. 한 출력 폴더에는 하나의 실행만 사용하세요.

자가대국은 실제 게임 결과를 매 위치의 둘 차례 관점으로 WDL 라벨에 기록합니다. 수 제한에 걸린 대국은 `TRUNCATED`로 기록하고 학습에서 제외합니다. 완료된 데이터가 하나도 없으면 세대 진행을 멈추고 원인을 출력합니다. 조기 기권이나 평가값 기반의 인공 승패 판정은 없습니다.

승격은 오프닝 쌍별 평균 점수에 대한 95% 단측 Hoeffding 하한이 0.55 이상일 때 수행합니다. 모든 대국이 완료되어야 하며 미완료 대국이 있으면 승격하지 않습니다. 작은 대결은 동작 확인용이며, 경기력 향상을 입증하지 못합니다. 이 판정은 Elo 추정이나 SPRT가 아닙니다.

개별 단계도 실행할 수 있습니다.

```powershell
.\run.ps1 selfplay --executable build\project_zero.exe --model runs\bootstrap\model.zero --output runs\games --games 8 --simulations 128
.\run.ps1 train --data 'runs/games/game_*.jsonl' --init runs\bootstrap\model.pt --output runs\candidate.pt --steps 1000
.\run.ps1 arena --executable build\project_zero.exe --candidate runs\candidate.zero --incumbent runs\bootstrap\model.zero --output runs\arena.json --pairs 64 --simulations 128
```

## 체크포인트 복구

```powershell
.\run.ps1 train --data 'runs/games/game_*.jsonl' --resume runs\candidate.pt --output runs\candidate.pt --steps 2000
.\run.ps1 export --checkpoint runs\candidate.pt --output models\candidate.zero
```

`--steps`는 총 목표 스텝입니다. `--resume`는 데이터 해시·순서, 배치 크기, 학습률, 장치, PyTorch 버전이 동일한 경우만 허용합니다. 모델, AdamW 모멘트, LR 진행 위치, Python/NumPy/Torch/CUDA RNG, 전용 샘플러 상태를 복구합니다. 데이터가 바뀌는 새 세대는 `--init`로 가중치만 가져옵니다.

저장은 임시 파일에 쓰고 flush/fsync한 뒤 교체합니다. 기존 정상 파일은 `.prev`로 보존합니다. 최신 파일을 읽을 수 없으면 `.prev`를 읽고 실제 사용한 경로를 결과에 기록합니다. 복구 후 export하면 복구된 가중치와 정확히 맞는 C++ 모델이 만들어집니다. 이 방식은 중단·부분 저장에 대한 복구를 제공하며, 전원 장애에 대한 파일시스템 수준의 완전한 트랜잭션을 보장하지는 않습니다.

## UCI

```text
uci
setoption name ModelFile value C:\path\to\model.zero
setoption name Simulations value 800
setoption name CPuct value 1.5
isready
position startpos moves e2e4 e7e5
go nodes 800
stop
quit
```

지원: `uci`, `isready`, `ucinewgame`, `position startpos/fen ... moves ...`, `setoption`, `go nodes/movetime/wtime/btime/winc/binc/movestogo/infinite`, `stop`, `quit`. `go depth N`은 호환용으로 `N*100` 시뮬레이션 예산에 매핑되므로 검증과 대결에는 `nodes` 또는 시간 제한을 사용하세요. `ponder`, `searchmoves`, Chess960, MultiPV는 제공하지 않습니다. `score cp`는 `400*(W-L)`의 표시용 값이며 보정된 센티폰 평가가 아닙니다.

`MaxNodes`는 보존할 트리 노드 수의 상한입니다. 상한 도달 후에는 새 노드를 저장하지 않고 잎을 평가합니다. 모델 파일이 없으면 명시적인 균등 prior / WDL=(0.25,0.5,0.25) 기준선으로 동작합니다. `ModelFile=<none>`으로 기준선을 선택할 수도 있습니다. 모델이 손상된 경우 이전에 로드한 정상 모델을 유지하고 오류를 출력합니다.

명령행 `project_zero.exe --model FILE`로 모델을 고정하거나 `--perft DEPTH "FEN"`으로 perft를 실행할 수 있습니다. 학습 도구용 `zero inspect`, `zero eval`, `zero search SIMULATIONS SEED NOISE TEMPERATURE`는 JSON 응답을 반환하는 확장 명령입니다.

## 검증 자료와 현재 한계

`validation/integration.json`과 `validation/trained-parity.json`에 자동 테스트 결과가 있습니다. 자세한 수치와 실험 산출물은 [VALIDATION.md](VALIDATION.md)에 정리했습니다. 전체 인코딩·탐색·모델 파일 명세는 [DESIGN.md](DESIGN.md)를 참고하세요.

현재 모델은 폭 64, residual 층 2개의 소형 MLP입니다. CPU에서 파이프라인을 검증하기 위한 모델이며, 강력한 체스 엔진에 필요한 학습량이나 표현력을 확보한 상태가 아닙니다. 실제 첫 자가대국 세대는 반복 무승부 3판과 중단 1판으로, 무승부로 쏠리는 문제도 관찰됐습니다. 대규모 학습을 시작하기 전에 데이터 다양성, 네트워크 구조, 전술 테스트를 개선해야 합니다.

다음 개발 순서: (1) 오프닝·전술·엔드게임 검증 세트와 수천 판 기준 대결, (2) CNN/ResNet과 배치 GPU 추론 백엔드, (3) 자가대국 여러 작업자와 스트리밍 replay, (4) 비트보드·make/unmake·트리 재사용, (5) 충분한 오프닝 쌍과 별도 홀드아웃에서 경기력 검증. 현재는 단일 탐색 스레드, 트리 재사용 없음, 배치 추론 없음, 테이블베이스 없음입니다.
