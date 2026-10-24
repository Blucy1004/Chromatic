# CHROMATIC ARC Shard Trainer

Intel Arc XPU와 CPU를 같은 Policy/WDL 모델·공식 Lichess 데이터로 비교하고 학습하는 별도 Windows 도구다. 기존 `chromatic-shard-trainer`, CHROMATIC 엔진, 메모리 회수 개선 엔진, Muyeong, 챔피언과 이전 실험 결과는 입력으로만 읽는다. 새 실행 데이터·체크포인트·프로파일은 **이 ARC 디렉터리 또는 사용자가 지정한 새 폴더**에 저장한다.

모델은 기존의 1,280 입력, 4,672 Policy 출력, WDL 출력 3개의 폭 64·잔차 블록 2 MLP다. `.zero` C++ 추론 형식은 유지한다. XPU를 무조건 빠르다고 가정하지 않으며, 측정 결과와 채택한 기본값은 [PERFORMANCE.md](PERFORMANCE.md)에 정리한다. 배치 연산만 비교한 결과는 [MICROBENCHMARK.md](docs/MICROBENCHMARK.md), 정확성·복구 시험 계약은 [VALIDATION.md](docs/VALIDATION.md)에 있다.

## 실행 환경

검증에 사용한 전용 Python은 다음 경로다.

```powershell
$Python = 'C:\Users\fly_t\chromatic-xpu\Scripts\python.exe'
& $Python --version
```

Windows x64, Python 3.11.9, PyTorch 2.14.1+xpu, NumPy 2.4.6 환경을 사용한다. GPU는 Intel Arc 140T이며 통합 GPU이므로 표시된 16GB를 독립적인 전용 VRAM이라고 해석하지 않는다. XPU 장치 인식뿐 아니라 실제 텐서 forward·backward·Optimizer 업데이트를 별도 검사한다. 기존 PyTorch 설치를 CPU 전용 패키지로 덮어쓰지 않는다. Python 패키지 설치나 큰 데이터 다운로드를 실행 스크립트가 자동으로 수행하지 않는다.

PowerShell의 Python 선택 순서는 `-Python`, `CHROMATIC_PYTHON`, 이 폴더의 `.venv`, 사용자 홈의 `chromatic-xpu`, 원래 작업 공간의 `../../work/.venv`, PATH의 `python`이다. 환경 선택을 확실히 하려면 `-Python $Python`을 명시한다. 아래 명령은 README가 있는 `chromatic-arc-trainer` 폴더에서 실행한다.

## 다운로드 없는 100개 기능 확인

먼저 설정만 검사한다. `-DryRun`은 파일·provenance·해시·옵션 범위·패키지 존재 여부를 확인하며 **PyTorch를 import하거나 XPU를 초기화하거나 다운로드·학습을 시작하지 않는다.** 실제 XPU 연산 지원은 본 실행과 수치 검사에서 확인한다.

```powershell
.\chromatic-train.ps1 -Python $Python -Positions 100 -ShardSize 25 -Epochs 3 `
  -Source .\fixtures\official-evaluations.jsonl.zst `
  -DataDirectory .\data\demo100 -CheckpointDirectory .\runs\demo100 `
  -Device xpu -Precision fp32 -MaxMemoryGB 4 -MaxDiskGB 1 -MaxDownloadMB 16 -DryRun
```

위 명령의 `-DryRun`을 빼면 로컬 공식 샘플로 실제 학습한다. 새 실험에는 DataDirectory와 CheckpointDirectory를 모두 새 폴더로 지정한다. 동일한 체크포인트 폴더를 다른 ledger가 공유할 수 없고, 기존 엔진·이전 학습기·이전 검증 폴더를 새 출력 경로로 지정하는 것도 거부한다.

100개 확인 후 같은 로컬 샘플에서 10,000개를 명시적으로 실행하는 예는 다음과 같다. 배치 256은 비교할 설정이며 기본 배치를 자동으로 바꾸는 명령이 아니다.

```powershell
.\chromatic-train.ps1 -Python $Python -Positions 10000 -ShardSize 1000 -Epochs 3 `
  -Source .\fixtures\official-evaluations.jsonl.zst `
  -DataDirectory .\data\arc10k -CheckpointDirectory .\runs\arc10k `
  -Device xpu -BatchSize 256 -Precision fp32 -CpuThreads 1 `
  -MaxMemoryGB 4 -MaxDiskGB 1 -MaxDownloadMB 16
```

`Positions`는 새 학습 포지션 수다. Epoch 반복·Replay 표본·고정 validation/test는 이 수에 더하지 않는다. 배포 샘플의 원본 행 60,000개가 모두 품질 검사를 통과하는 학습 포지션은 아니다. 목표·품질·균형 조건을 충족하지 못하면 명확히 중단하며 확정된 상태와 미완료 샤드를 보존한다.

## CPU, XPU와 자동 선택

현재 기본값은 `-Device auto -CpuThreads 1 -BatchSize 32 -Precision fp32 -CacheMode auto -Prefetch 0 -ValidationMode selected`다. 선택된 평가의 실제 PV는 끝까지 검증하면서 사용하지 않는 낮은 평가의 중복 검증을 줄이는 selected 경로를 60,000행 의미 비교·동일 데이터 학습·CPU/XPU 중단 복구로 확인한 뒤 채택했다. 전송은 `copy`, Optimizer는 `default`, gradient accumulation은 1이다. BF16·GPU 상주 캐시·fused Optimizer·배치 확대·compile을 기본값으로 강제하지 않는다.

```powershell
# 같은 데이터의 CPU 기준 조건
.\chromatic-train.ps1 -Python $Python -Positions 1000 -ShardSize 250 -Epochs 3 `
  -Source .\fixtures\official-evaluations.jsonl.zst `
  -DataDirectory .\data\cpu1k -CheckpointDirectory .\runs\cpu1k `
  -Device cpu -CpuThreads 1 -BatchSize 32 -Precision fp32 -MaxDownloadMB 16
```

`auto`는 모델 구조·입출력 크기·PyTorch·운영체제·CPU·GPU 정보를 검사하고, source·초기 모델·학습 규모·Epochs·Replay·업데이트 수·배치·CPU 스레드까지 일치하는 **검증된 전체 파이프라인 결과**를 먼저 사용한다. 해당 범위가 일치하지 않으면 그 자료를 다른 규모로 외삽하지 않고 마이크로벤치마크 또는 짧은 CPU/XPU 실측으로 돌아간다. 측정 근거는 [등록된 추천](reports/benchmarks/pipeline-recommendation.json)에 있다. 사용자가 지정한 BatchSize나 CpuThreads를 조용히 바꾸지 않는다. 선택 근거는 CheckpointDirectory의 `device-recommendation.json`에 남는다. 최대 처리량 배치 추천과 학습 품질을 유지하는 기본 배치는 별개다.

예를 들어 100개·1에폭의 기본 auto 확인은 실제 CPU를 선택해 4업데이트로 실행됐지만, 그 작은 실행에 10,000개·3에폭의 가속 배율을 적용하지 않는다. [실제 PowerShell 실행 증거](reports/powershell-auto-validation.json)와 요청 규모·배치가 등록 범위와 다르면 해당 추천을 거부하는 [자동 선택 함수 검증](reports/auto-selection-validation.json)을 구분해 기록했다.

같은 환경에서 기존 CPU1 학습기를 2회 재현한 10,000개 전체 시간은 82.25~105.29초, 중앙값 93.77초였다. 최종 selected 경로의 CPU4·batch32는 3회 중앙값 58.84초로 이 기준의 1.59배였다. 과거 Python 3.12의 132.6초 대비 2.25배는 환경과 시점이 다른 참고 비교다. Arc FP32는 3회 중앙값 71.28초로 동작했으며 이 작은 모델의 전체 처리량 최선은 CPU4였다. 상세 범위·자원·손실은 [PERFORMANCE.md](PERFORMANCE.md)를 확인한다.

검증한 CPU4 조건을 새 실행으로 재현하려면 다음 명령을 사용한다. 실행을 원할 때 사용자가 직접 시작하는 명령이다.

```powershell
.\chromatic-train.ps1 -Python $Python -Positions 10000 -ShardSize 1000 -Epochs 3 `
  -Source .\fixtures\official-evaluations.jsonl.zst `
  -DataDirectory .\data\verified-cpu4 -CheckpointDirectory .\runs\verified-cpu4 `
  -Device cpu -CpuThreads 4 -BatchSize 32 -Precision fp32 -ValidationMode selected `
  -ReplaySize 256 -CheckpointEvery 32 -Seed 2026 `
  -MaxMemoryGB 4 -MaxDiskGB 1 -MaxDownloadMB 16
```

명시적인 `-Device xpu`는 실제 작은 forward/backward가 실패하면 기본적으로 오류를 낸다. 사용자가 `-Fallback cpu`를 지정했을 때만 초기 장치 선택 실패를 CPU FP32로 전환한다. 임의의 학습 중 장치 오류 뒤에 부분 업데이트를 유지한 채 조용히 CPU로 넘어가지 않는다. XPU BF16은 `-Device xpu -Precision bf16`로 선택하며, 손실/gradient 비정상 검출 시 Optimizer가 실행되기 전 해당 배치를 버리고 FP32로 다시 계산한다. FP32에서도 NaN/Inf가 발생하면 마지막 정상 체크포인트와 샤드를 보존하고 중단한다.

BF16은 처음과 매 32번째 업데이트에서 같은 배치의 FP32 손실을 별도로 비교한다. BF16 손실 악화가 `max(0.02, FP32 손실의 2%)`를 넘으면 해당 업데이트 전 gradient를 버리고 FP32로 복귀한다. 이 작은 배치 검사와 별개로 장기 검증 손실·체스 품질도 확인해야 한다.

## 지원 옵션

기존의 `./chromatic-train.ps1 -Positions 100000 -ShardSize 10000 -Epochs 3` 형식을 유지한다. 이 명령은 `Source`를 생략하므로 공식 온라인 원본을 사용한다. 이번 작업의 소규모 검증을 넘어서는 큰 다운로드나 100,000개 실험을 자동 시작하는 의미는 아니다.

|옵션|기본값|의미|
|---|---:|---|
|`-Device`|`auto`|`auto`, `cpu`, `xpu`|
|`-CpuThreads`|1|PyTorch CPU 스레드 수, 실제 논리 CPU 수 이내|
|`-Precision`|`fp32`|`fp32`, XPU 실험용 `bf16`|
|`-BatchSize`|32|Replay를 포함한 유효 배치, 최대 4096|
|`-Prefetch`|0|0~16, source 행과 CPU 배치의 순서 보장 bounded prefetch 깊이|
|`-CacheMode`|`auto`|`auto`, `none`, `memory`, `mmap`, 명시적 `device`|
|`-ValidationMode`|`selected`|`selected`: 선택 평가의 전체 PV 검증, `full`: 기존 모든 평가 검사·거부 감사 범위|
|`-TransferMode`|`copy`|`copy`, XPU 전송 버퍼 재사용 실험용 `preallocated`|
|`-GradientAccumulation`|1|유효 배치를 나눠 backward하는 micro-batch 수|
|`-Optimizer`|`default`|AdamW의 `default`, `loop`, `foreach`, `fused` 비교|
|`-Fallback`|`error`|장치 초기화 실패 시 `error` 또는 명시적 `cpu`|
|`-Benchmark`|꺼짐|로컬 고정 데이터로 장치/배치 연산 행렬 실험, 학습 후보 저장 없음|
|`-Profile`|꺼짐|단계별 CPU/GPU 동기화 계측과 상세 stage 로그|
|`-Positions`|100000|새 학습 포지션 목표|
|`-ShardSize`|10000|샤드 크기, 최대 100000|
|`-Epochs`|3|샤드별 반복 수|
|`-LearningRate`|0.001|초기 AdamW 학습률, 0보다 크고 1 이하|
|`-DataDirectory`|`data/`|ledger·고정 세트·샤드·binary cache·일시 spool|
|`-CheckpointDirectory`|`runs/default/`|후보·체크포인트·로그·자원/성능 기록|
|`-Resume`|꺼짐|저장한 설정과 확정된 학습 상태로 재개|
|`-DryRun`|꺼짐|장치 초기화·학습·다운로드 없는 설정 검사|
|`-KeepShards`|꺼짐|성공 샤드와 해당 정규화 자료 보관|
|`-MaxDiskGB`|4|생성 데이터와 체크포인트 폴더 합산 사용량 예산|
|`-MaxMemoryGB`|4|Windows 프로세스 committed memory 하드 한도와 RSS 검사|
|`-Seed`|2026|분할·순열·Replay·학습 재현성 시드|
|`-Source`|공식 eval URL|공식 HTTPS 또는 provenance와 해시가 일치하는 로컬 JSONL/Zstandard|
|`-Calibration`|`fixtures/calibration.json`|독립 결과로 보정한 CP→WDL artifact|
|`-InitialCheckpoint`|`champion/model.pt`|새 실험의 모델 구조와 시작 가중치|
|`-MaxDownloadMB`|128|프로세스의 byte-0 기준 압축 입력 범위 상한|
|`-CheckpointEvery`|32|정상 학습 중 저장하는 Optimizer 단계 간격|
|`-ReplaySize`|512|체크포인트에 담는 대표 과거 표본 상한|
|`-Python`|자동 선택|실행할 Python 경로|

GB는 1024³바이트, MB는 1024²바이트다. `MaxDownloadMB`는 재시도를 합친 전체 실행의 누적 전송량 상한이 아니며 재개 시 같은 압축 접두부를 다시 읽을 수 있다. Python CLI에는 `--min-depth 12`, `--min-knodes 10`, `--similarity-cap 50`, `--openings-directory`, `--benchmark-output`도 있다. 전체 형식은 `& $Python -m shardtrain.run --help`로 확인한다. `torch.compile`은 현재 실행기의 기본 학습 경로에 넣지 않았고 별도 격리 실험으로 보고한다.

## 중단과 재개, 장치 간 이동

```powershell
.\chromatic-train.ps1 -Python $Python -Resume -DataDirectory .\data\arc10k
```

원래 설정·장치·정밀도·배치·샤드 크기·Optimizer 조건을 복원한다. 체크포인트는 CPU 텐서로 저장하며 모델·Optimizer·Scheduler·CPU 및 XPU RNG·Replay·순열·에폭·offset·소스 cursor를 포함한다. 저장 완료 상태는 SQLite 단일 트랜잭션으로 확정한다. 본체 손상 시 미러, 둘 다 손상되면 필요한 보존 샤드가 남은 이전 정상 체크포인트를 검토한다. 실패한 샤드는 삭제하지 않으며 성공한 샤드도 마지막 한 개는 rollback을 위해 남긴다.

ValidationMode가 없던 이전 ARC 실행을 재개하면 `full`을 자동 복원한다. 새 기본값 selected로 기존 실행의 처리 의미를 바꾸지 않으며, full 서명에서 이 키를 생략하는 이전 계약도 유지한다. 진행 중인 full 실행을 selected로 바꾸는 대신 다른 새 실행으로 시험한다.

같은 ARC 실행의 장치를 바꾸는 Resume는 설정 불일치로 거부한다. 장치 간 PT 저장 호환성과 같은 장치 중단 재현성은 구분한다. 기존 CPU PT나 새 XPU PT를 다른 장치의 새 실행에 사용하려면 **새 DataDirectory·CheckpointDirectory와 `-InitialCheckpoint <PT>`**를 지정한다. 이 경로는 모델 가중치에서 새 Optimizer로 시작하며 과거 실행의 학습 위치를 이어가는 기능이 아니다. 기존 CPU 학습기는 기존 경로에서 그대로 사용할 수 있다.

기존 CPU 샤드 실행의 **Optimizer·Scheduler·RNG·Replay·샤드 진행 전체**를 보존해야 한다면 [Legacy import 절차](docs/LEGACY_IMPORT.md)를 사용한다. 원본 SQLite를 immutable 읽기 전용으로 검증하고 새 빈 디렉터리에 복사한 뒤, 새 복사본만 CPU FP32의 동일 학습 계약으로 ARC에 이행한다. 기존 실행을 직접 수정하거나 XPU로 설정을 바꾼 Resume를 허용하는 기능은 아니다.

원본·ledger·manifest에는 절대 경로와 원본 파일 정체성이 기록되므로 기존 실행을 재개하려면 원래 경로를 유지한다. 다른 위치로 옮긴 배포본은 새 실행을 시작할 수 있지만 이전 실행의 경로를 자동 재작성하지 않는다. 완전한 ledger·모든 정상 사본·필요 샤드를 모두 잃었을 때 복구할 수 있다고 주장하지 않는다.

## 데이터 품질과 캐시

공식 [Lichess Evaluation Database](https://database.lichess.org/#evals)의 CC0 JSONL/Zstandard를 사용한다. URL·ETag/버전·라이선스·행 번호·접두부 SHA를 기록하고 압축 원본 전체를 RAM이나 디스크에 받아 놓는 것을 요구하지 않는다. 임의 중간 압축 바이트를 새 스트림처럼 해석하지 않으며, 새 프로세스는 byte 0부터 재생해 저장한 행 prefix 해시를 검증한다. 따라서 후반부 재시작에는 앞부분 읽기 비용이 있다.

FEN 합법성·평가 깊이·노드 수·PV 합법성·정규화 중복·유사 구조 집중 제한·family 단위 분할을 유지한다. 평가 원본의 4필드 FEN에는 실제 ply와 반복 이력이 없으므로 이를 꾸며내지 않는다. 별도 공식 CC0 오프닝 이력으로 시작 위치와 0~12플라이를 정책 앵커에 포함하며, 앵커의 Value 가중치는 0이다. 단일/복수 PV는 관측된 부분 정책이고 낮은 신뢰도의 모방 자료다. CP는 백 관점에서 현재 차례로 정규화하고, 별도 완료 게임 결과로 보정한 약한 WDL을 사용한다. 이 WDL은 완벽한 플레이의 이론적 승률이 아니다.

고정 validation/test는 각각 `max(16, min(1000, Positions // 10))`개를 최초에 만들고 샤드가 바뀌어도 유지한다. 보정·전술 family는 신규 학습에서 제외한다. 작은 실험 간 loss를 공정하게 비교할 때는 동일한 고정 세트를 별도로 적용해야 한다. Validation은 FP32, 배치 256으로 실행한다. 기존 배치 32 검증과의 총합 마지막 자릿수에는 부동소수점 합산 차이가 있을 수 있다.

현재 기본 selected와 full은 공식 60,000행에서 선택된 평가·Policy/WDL·가중치·FEN/history·identity·provenance 의미가 일치했다. selected는 사용하지 않는 낮은 평가의 세부 rejection-audit 범위가 다르므로 그 감사 목록까지 같다고 주장하지 않는다. 원본 전체 평가를 같은 범위로 검사해야 하면 `-ValidationMode full`로 새 실행을 시작한다. [정규화 의미 비교](reports/quality-selected/validation.json)에 제외한 감사 필드와 0 mismatch 근거를 기록했다.

`auto` 캐시는 검증된 현재 샤드·Replay·고정 세트를 compact binary로 인코딩해 작은 것은 제한된 CPU RAM에, 큰 것은 read-only mmap에 둔다. binary에는 1,280개 feature, bit-packed 합법수 mask, sparse 정책과 confidence/WDL을 저장한다. dense Policy/합법수 텐서는 요청한 배치에만 만든다. `none`은 참조 인코더 경로, `memory`는 명시적 RAM 캐시이며 예산 초과 시 오류다. `device`는 XPU의 현재 샤드+Replay dense 상주 캐시를 명시적으로 요청하는 모드로, 별도 한도를 넘으면 `auto`/`mmap` 사용을 안내한다. 기본값에서 전체 데이터셋을 RAM이나 XPU에 올리지 않는다. 세부 계약과 파일 구조는 [DATA_PIPELINE.md](docs/DATA_PIPELINE.md)에 있다.

디스크 예산은 DataDirectory와 CheckpointDirectory를 중복 없이 합산한다. Python 환경, 다른 곳에 둔 입력 fixture, 챔피언·기존 엔진은 예산 밖이다. SQLite의 영구 중복 키는 학습량에 따라 증가한다. RAM 한도는 프로세스 전체 committed memory와 RSS에 적용되며 매우 작은 한도나 큰 샤드·배치는 중단될 수 있다. 통합 GPU의 allocator bytes와 프로세스 RAM을 단순히 합산해 물리 사용량으로 표시하지 않는다.

## 벤치마크와 계측

```powershell
.\chromatic-train.ps1 -Python $Python -Benchmark
```

위 단축 명령은 패키지에 포함된 `fixtures/benchmark-records.jsonl`의 고정 1,000행으로 인코딩 배치 행렬을 측정한다. 원본은 읽기만 하고 다운로드·샤드 학습·챔피언 덮어쓰기를 하지 않는다. 다른 고정 세트를 사용하려면 직접 정규화 JSONL을 지정한다.

```powershell
& $Python .\tools\benchmark_device.py --records <고정-test.jsonl> `
  --output .\reports\benchmarks --repeats 3 --steps 15 --warmup 2 --deadline-seconds 240
```

CPU 1/2/4/8, XPU FP32/BF16, 배치 32/64/128/256/512와 전송·상주·Optimizer·누적 기법을 비교하고 중앙값·실패 이유·추천을 CSV/JSONL/JSON에 저장한다. 이는 이미 인코딩한 배치 연산 실험이다. 데이터 준비·저장·검증까지 포함한 전체 처리량은 [PERFORMANCE.md](PERFORMANCE.md)의 별도 실험과 구분한다.

일반 실행에 `-Profile`을 붙이면 CheckpointDirectory의 `profile/stages.jsonl`, `profile/profile.json`에 세부 단계가 기록된다. GPU 단계는 `torch.xpu.synchronize()`를 적용해 enqueue 시간으로 오인하지 않도록 하며, 이 동기화 비용은 계측 실행에만 추가한다. CPU 단계는 XPU를 동기화하지 않는다. 전체 성능 비교에는 Profile을 끈 실행을 사용하고, 단계 진단용 Profile 실행을 별도로 표시한다.

자원은 `resources/resources.csv`, `resources/resources.jsonl`, `resources/resource-summary.json`에 약 0.2초 간격으로 기록한다. CPU process%는 여러 코어를 쓰면 100%를 넘을 수 있다. GPU%는 Windows PDH의 **현재 PID에서 가장 바쁜 GPU engine**이며 EU occupancy가 아니다. 드라이버·권한·counter warmup 때문에 얻지 못한 값은 NA와 이유로 남기며 0%라고 주장하지 않는다. 디스크 I/O는 프로세스 카운터이며 물리 드라이브 트래픽과 같지 않다.

`training-report.json`의 `execution`은 순수 학습 구간·전송 구간·학습 샘플 수·samples/sec를 따로 담는다. 학습 샘플 수에는 Epoch와 Replay가 포함된다. 순수 학습 구간에는 forward/backward/Optimizer뿐 아니라 gradient clip과 finite 검사도 포함한다. `process_seconds`는 Trainer 생성 이후의 구간이며 Python 시작·사전 auto 선택·오프닝 목록 준비는 제외될 수 있으므로, 전체 실행 시간은 벤치마크 런처의 외부 wall clock을 기준으로 본다.

## 확인 도구와 결과 보존

```powershell
& $Python -m pytest -q
& $Python .\tools\validate_arc.py --help
```

단위 검사는 작은 로컬 자료로 캐시 동일성·손상 거부·prefetch 순서와 cursor·RNG·분할·복구·계측을 확인한다. 실제 XPU 수치/역전파/업데이트, 독립 C++ `.zero` 일치, 실제 강제 종료·Resume는 별도 [검증 계약](docs/VALIDATION.md)의 제한된 명령으로 실행하고 원시 JSON에 기록한다. 같은 장치·같은 설정의 중단/재개는 정확 비교하며, CPU↔XPU는 미리 정한 허용 오차로 비교한다.

새 학습 모델은 항상 후보다. loss 감소나 encoded-batch 속도만으로 챔피언에 승격하지 않고 자동 덮어쓰기 경로도 제공하지 않는다. `.zero` 내보내기·독립 고정 세트·수치 안정성·전술·충분한 완료 대국 근거가 필요하다. 기존 엔진 및 챔피언 해시는 보존 보고서로 대조한다. 프로그램 라이선스는 [LICENSE](LICENSE), 외부 구성요소는 [THIRD_PARTY.md](THIRD_PARTY.md), 공식 데이터별 provenance는 `fixtures/*.source.json`을 확인한다.

이번 후보의 편의 사본은 아래에 제공한다. PT는 새 실행의 InitialCheckpoint, `.zero`는 호환 C++ 엔진에서 모델을 명시적으로 골라 시험하는 용도다. 어느 파일도 기존 챔피언에 자동 적용되지 않는다.

|후보|학습용 PT|C++ 추론 가중치|
|---|---|---|
|CPU4 selected|[PT](candidates/cpu4-selected.pt)|[.zero](candidates/cpu4-selected.zero)|
|XPU FP32 selected|[PT](candidates/xpu-fp32-selected.pt)|[.zero](candidates/xpu-fp32-selected.zero)|
|XPU BF16|[PT](candidates/xpu-bf16.pt)|[.zero](candidates/xpu-bf16.zero)|

최종 Python 3.11 일반 검사 151개와 24개 subtest가 통과했고, 별도 실제 GPU·복구·내보내기 시험도 기록했다. 6개 선택 실행 테스트는 일반 검사에서 생략되고 해당 전용 시험으로 따로 확인한다. mixed 입력은 Policy/WDL 양쪽 gradient와 업데이트를 실제로 만들었고 FP32 CPU/XPU 가중치 한 단계 오차는 최대 약 1.56e−7이었다. 독립 C++ 추론은 CPU4-selected·XPU-FP32-selected·XPU-BF16 후보 각각 10개, 총 30개가 통과했다. [수치 결과](reports/numeric-xpu-fp32.json), [C++ 결과](reports/native-summary.json), [selected 복구 결과](reports/recovery-selected-summary.json), [기존 실행 실제 import 결과](reports/legacy-import-validation.json)를 확인한다.

공식 표본의 수집·원본 프레임 검증 기록은 보존된 이전 작업의 증거를 바이트 그대로 이어받았다. 이를 ARC에서 새 원격 다운로드나 새 소스 코드 회귀 시험을 실행한 결과로 재표시하지 않는다. 출처 이력과 이번 ARC 결과의 범위는 [SOURCES.md](docs/SOURCES.md)를 따른다.
