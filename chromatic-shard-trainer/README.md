# CHROMATIC Lichess Shard Trainer

공식 Lichess CC0 평가 데이터를 순차적으로 읽고, 제한된 크기의 샤드를 학습·검증·저장한 뒤 교체하는 Windows PowerShell용 학습 도구입니다. 기존 CHROMATIC의 PyTorch Policy/WDL 모델과 인코딩을 별도 `vendor/` 사본으로 사용합니다. 기존 CHROMATIC·메모리 개선 엔진·Muyeong 소스와 챔피언 모델은 변경하지 않습니다.

이 버전은 **CPU 학습, PyTorch 스레드 1개**를 사용합니다. 처음 시작할 때 `champion/model.pt`의 모델 구조와 가중치를 읽고 **새 AdamW Optimizer·학습률 스케줄러·학습 단계 0**으로 시작합니다. `-Resume`에서는 이 도구가 저장한 모델뿐 아니라 Optimizer, Scheduler, RNG, 샤드 및 배치 위치도 복원합니다.

공식 데이터로 100·1,000·10,000개 포지션 학습과 실제 프로세스 강제 종료 후 복구를 검증했습니다. 100,000개 이상 학습은 구현된 실행 범위이며 이번 검증에서 실행하지 않았습니다. 결과와 한계는 [실험 보고서](reports/EXPERIMENTS.md)를 확인하세요.

## 1. 설치

아래 명령은 이 README가 있는 `chromatic-shard-trainer` 폴더에서 실행합니다. 검증 환경은 Windows PowerShell **5.1**, Python **3.12.14**, CPU용 PyTorch입니다. 별도 가상환경을 사용하며 의존성 설치는 사용자가 직접 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

설치는 Python 패키지를 받습니다. 학습 데이터 다운로드나 모델 학습을 시작하지 않습니다. 고정된 의존성 버전은 [requirements.txt](requirements.txt)에 있습니다. 기존 환경을 쓰려면 실행 시 `-Python "C:\경로\python.exe"`를 지정할 수 있습니다.

PowerShell 래퍼의 Python 선택 순서는 `-Python`, `CHROMATIC_PYTHON` 환경변수, 이 폴더의 `.venv`, 원래 작업 공간의 `../../work/.venv`, PATH의 `python`입니다. 새 환경에서는 위의 로컬 `.venv` 설치를 사용하세요.

## 2. 먼저 다운로드 없는 작은 실행

배포된 `fixtures/official-evaluations.jsonl.zst`는 공식 데이터 앞부분에서 가져온 60,000개 JSONL 레코드입니다. 별도 `.source.json`에 원본 URL·ETag·CC0·프레임 검증·로컬 SHA-256이 기록되어 있습니다. **60,000개 원본 레코드가 모두 학습 가능한 포지션이라는 뜻은 아닙니다.** 중복·품질 검사·독립 데이터 제외·단계별 균형 선별 후 필요한 수만 학습합니다.

먼저 설정을 검사합니다.

```powershell
.\chromatic-train.ps1 -Positions 100 -ShardSize 25 -Epochs 1 `
  -Source .\fixtures\official-evaluations.jsonl.zst `
  -DataDirectory .\data\demo100 -CheckpointDirectory .\runs\demo100 `
  -MaxDownloadMB 16 -MaxDiskGB 0.5 -MaxMemoryGB 2 -DryRun
```

`-DryRun`은 옵션 범위, 파일 존재, 로컬 원본의 provenance와 해시, 의존성 유무를 검사합니다. 다운로드·학습·샤드 삭제는 수행하지 않습니다. 전체 데이터 파싱, 모델 텐서 로딩, 실제 자원 적합성 검증은 본 실행에서 이뤄집니다.

실제 100개 학습도 한 명령으로 실행합니다.

```powershell
.\chromatic-train.ps1 -Positions 100 -ShardSize 25 -Epochs 1 `
  -Source .\fixtures\official-evaluations.jsonl.zst `
  -DataDirectory .\data\demo100 -CheckpointDirectory .\runs\demo100 `
  -MaxDownloadMB 16 -MaxDiskGB 0.5 -MaxMemoryGB 2
```

이 명령은 로컬 공식 표본을 사용하므로 네트워크 요청이 없습니다. 새 실험에는 **DataDirectory와 CheckpointDirectory를 둘 다 새 폴더로** 지정하세요. 기존 실행을 다시 지정하면 `-Resume`을 요구합니다. 여러 데이터 폴더가 같은 체크포인트 폴더를 공유하는 것도 거부합니다.

`Positions`는 시작 위치와 오프닝 앵커를 포함한 **새 학습 포지션의 목표 수**입니다. Epoch 반복, Replay 재사용, validation/test 포지션은 여기에 더하지 않습니다. 초반·중반·종반은 각 샤드에서 거의 1:1:1로 선별하며 나머지 1~2개를 순서대로 배분합니다.

## 3. 중단과 재개

Ctrl+C나 프로세스 종료 후 다음 명령을 사용합니다.

```powershell
.\chromatic-train.ps1 -Resume -DataDirectory .\data\demo100
```

저장된 설정에서 원본, 체크포인트 폴더, Positions, Epochs, BatchSize, 학습률, Seed 등을 자동으로 복원합니다. 마지막으로 SQLite에 **확정된 체크포인트**의 배치 위치부터 재개합니다. 그 이후 메모리에서만 실행한 배치는 다시 계산할 수 있지만 확정된 진행을 중복 반영하지 않습니다. 이미 목표를 달성한 실행은 학습 단계를 추가하지 않습니다.

재개 시 디스크·메모리·입력 읽기 예산과 샤드 보존 여부는 명시적으로 조정할 수 있습니다.

```powershell
.\chromatic-train.ps1 -Resume -DataDirectory .\data\demo100 `
  -MaxDiskGB 1 -MaxMemoryGB 2 -MaxDownloadMB 32
```

학습 목표, 모델, 학습률, 배치·샤드 크기, Epochs, Seed, 품질 기준, 원본과 보정 파일을 바꾸는 재개는 거부합니다. 새 설정을 시험하려면 새 데이터·체크포인트 폴더로 시작하세요. `-KeepShards:$false`는 재개 시 샤드 보존을 해제하는 PowerShell 표기입니다.

**진행 중인 실행은 원래 절대 경로를 유지해야 합니다.** 설정, 원본 정체성, 고정 데이터와 manifest에 절대 경로가 저장됩니다. 현재 버전은 실행 폴더를 다른 위치·PC로 옮긴 뒤 경로를 재작성하는 기능을 제공하지 않습니다. 백업은 학습을 멈춘 상태에서 데이터 폴더와 체크포인트 폴더를 함께 보관하고, 재개하려면 원래 경로로 복원하세요. Python 패키지 버전과 CPU 실행 조건도 유지하는 편이 재현성 검증에 유리합니다.

## 4. 공식 온라인 원본을 쓰는 명령

다음은 사용자가 대규모 실험을 선택했을 때 실행할 수 있는 명령 예입니다. **이번 작업에서 이 100,000개 학습을 시작하지 않았습니다.** `-Source`를 생략하면 공식 평가 데이터 URL을 사용합니다.

```powershell
.\chromatic-train.ps1 -Positions 100000 -ShardSize 10000 -Epochs 3 `
  -DataDirectory .\data\lichess100k -CheckpointDirectory .\runs\lichess100k `
  -MaxDownloadMB 128 -MaxDiskGB 4 -MaxMemoryGB 4
```

공식 전체 압축 파일을 먼저 내려받지 않고 앞부분의 완성된 Zstandard 프레임을 처리합니다. 품질·중복·균형 조건에 따라 읽어야 할 원본 레코드 수가 달라지므로 `MaxDownloadMB=128`이 목표 수를 반드시 충족한다는 보장은 없습니다. 입력 예산이나 단계별 표본이 부족하면 명확히 중단하고 확정 데이터와 진행을 보존합니다. 필요한 경우 사용자가 예산을 검토한 뒤 `-Resume -MaxDownloadMB ...`로 늘릴 수 있습니다.

`MaxDownloadMB`는 프로세스마다 원본의 byte 0부터 읽는 **압축/입력 바이트 범위의 상한**입니다. 전체 학습의 누적 네트워크 전송량 상한은 아닙니다. 연결 재시도와 프로세스 재개는 같은 앞부분을 다시 전송할 수 있습니다. 임의의 중간 압축 바이트를 독립 스트림으로 취급하지 않습니다. 상세 동작은 [스트리밍 계약](docs/STREAMING.md)에 있습니다.

## 5. 지원 옵션

| PowerShell 옵션 | 기본값 | 의미 |
|---|---:|---|
| `-Positions` | 100000 | 새 학습 포지션 목표 수, 앵커 포함 |
| `-ShardSize` | 10000 | 샤드당 최대 포지션, 최대 100000 |
| `-Epochs` | 3 | 각 샤드 반복 횟수 |
| `-BatchSize` | 32 | Replay를 포함한 미니배치 크기, 최대 4096 |
| `-LearningRate` | 0.001 | 새 실행의 AdamW 초기 학습률, 0보다 크고 1 이하 |
| `-DataDirectory` | `data/` | SQLite ledger, 고정 세트, 샤드, 디코드 임시 파일 |
| `-CheckpointDirectory` | `runs/default/` | 후보 체크포인트, 로그와 보고서 |
| `-Resume` | 꺼짐 | 저장된 설정과 확정 상태로 재개 |
| `-DryRun` | 꺼짐 | 설정 검사만 실행 |
| `-KeepShards` | 꺼짐 | 완료 샤드와 해당 정규화 payload도 보관 |
| `-MaxDiskGB` | 4 | 데이터·체크포인트 폴더 합산 사용량 예산, 최소 0.1 |
| `-MaxMemoryGB` | 4 | 학습 프로세스 메모리 한도, 최소 0.5 |
| `-Seed` | 2026 | 분할·샘플 순서·학습 RNG 시드 |
| `-Source` | 공식 평가 URL | 공식 HTTPS 원본 또는 provenance가 있는 로컬 JSONL/Zstandard |
| `-Calibration` | `fixtures/calibration.json` | 독립 대국 결과로 보정한 CP→WDL 파일 |
| `-InitialCheckpoint` | `champion/model.pt` | 새 실행의 초기 모델 구조와 가중치 |
| `-MaxDownloadMB` | 128 | 입력 압축/파일 바이트 범위 상한, MiB 단위 |
| `-CheckpointEvery` | 32 | 정상 학습 중 저장하는 Optimizer 단계 간격 |
| `-ReplaySize` | 512 | 체크포인트에 포함하는 대표 과거 학습 샘플 수 상한 |
| `-Python` | 자동 선택 | 실행할 Python 경로 |

`GB`는 1024³바이트, `MB`는 1024²바이트로 계산합니다. 양수·유한값이어야 합니다. `CheckpointEvery`와 별개로 초기화, 샤드 시작, 샤드 완료 시에도 저장합니다.

Python 모듈은 같은 옵션을 `--positions`, `--shard-size` 같은 kebab-case로 제공합니다. 다음 품질 옵션은 **Python CLI에서만** 제공합니다.

| Python 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--min-depth` | 12 | 최소 교사 평가 깊이 |
| `--min-knodes` | 10 | 최소 교사 평가 kilo-node 수 |
| `--similarity-cap` | 50 | 동일 기물 수·폰 구조 그룹의 전역 집중 상한 |
| `--openings-directory` | `fixtures/` | 고정된 공식 오프닝 TSV 및 provenance 위치 |

```powershell
.\.venv\Scripts\python.exe -m shardtrain.run --help
```

GPU·학습 스레드 수를 바꾸는 옵션은 이 버전에 없습니다.

## 6. 데이터와 라벨의 의미

- 공식 평가 데이터는 [Lichess Evaluation Database](https://database.lichess.org/#evals)의 CC0 JSONL/Zstandard만 허용합니다. 로컬 파일에는 정확한 공식 URL, `CC0-1.0`, 로컬 SHA-256이 맞는 `<파일명>.source.json`이 필요합니다.
- FEN 합법성, 깊이·노드 수, PV 첫 수와 이후 수의 합법성을 검사합니다. 동일 법적 위치의 중복을 제거하고 유사 폰 구조의 과다 편중을 제한합니다. 파싱·품질 거부 이유와 수를 ledger에 남깁니다.
- 평가 데이터의 4필드 FEN에는 실제 수순·50수 카운터·반복 이력이 없습니다. 초반/중반/종반 구분은 기물 수 기반 대용 지표입니다. 실제 첫 6~12플라이 노출은 [공식 CC0 오프닝 카탈로그](https://github.com/lichess-org/chess-openings)의 이력이 있는 약한 Policy 앵커로 보완합니다. 시작 위치는 학습에 반드시 포함하고 앵커 Value 가중치는 0입니다.
- CP와 mate는 백 관점에서 읽고 현재 둘 차례의 관점으로 정규화합니다. WDL 순서는 **현재 차례의 승·무·패**입니다. CP를 승률로 직접 간주하지 않습니다. 보정 파일이 수용되지 않으면 CP Value 가중치는 0이며 임시 WDL 값이 실제 무승부 라벨로 학습되지 않습니다.
- 여러 PV의 관측된 첫 수에만 부분 Policy를 만들고, 단일 PV는 낮은 가중치의 모방 라벨로 취급합니다. 완전한 탐색 방문 분포나 관측하지 않은 수의 열세를 증명하는 데이터가 아닙니다. mate도 부호에 따른 약한 교사 라벨입니다.
- CP→WDL 보정에는 완료된 별도 공식 PGN의 실제 결과를 사용합니다. 게임 단위로 fit/validation/test를 분리하고 마지막 test의 NLL·Brier·ECE를 보고합니다. 이는 인간 대국의 경험적 결과 보정이며 완벽한 플레이의 이론적 WDL이 아닙니다.
- 학습/검증/테스트는 유사 구조 그룹 단위의 시드 고정 분할입니다. 기물 배치가 같은 가족이 분할을 넘지 않도록 영속적으로 확인합니다. validation/test는 각각 `max(16, min(1000, Positions // 10))`개를 최초 생성 후 고정합니다. 보정·전술 평가의 가족은 신경망 학습에서 제외합니다.

자세한 라벨 가중치와 보정 근거는 [LABELS_AND_CALIBRATION.md](docs/LABELS_AND_CALIBRATION.md), 영속화 구조는 [ARCHITECTURE.md](docs/ARCHITECTURE.md)에 있습니다.

## 7. 자원 사용과 저장 결과

압축 파일이나 디코드된 전체 데이터를 RAM에 올리지 않습니다. 한 Zstandard 프레임을 작은 청크로 임시 디스크에 디코드하고 프레임 검증이 끝난 뒤 레코드를 제공합니다. 프레임 임시 파일 상한 128MiB, Zstandard window 상한 128MiB, JSONL 한 줄 상한 1MiB를 적용합니다. 전체 디코드 상한은 `max(128, MaxDownloadMB × 16)`MiB입니다. 입력 예산을 늘려도 프레임·window·한 줄 상한은 자동으로 커지지 않습니다.

디스크 예산은 **DataDirectory와 CheckpointDirectory 아래의 합산**입니다. 둘 중 하나가 다른 하나의 하위면 중복 합산하지 않습니다. Python 환경, 원본 fixture를 다른 곳에 둔 경우 그 파일, 보존된 챔피언, 기존 엔진은 예산 밖입니다. SQLite 중복 키·가족 기록은 샤드를 삭제해도 남으므로 전체 학습량에 따라 디스크가 증가합니다. 이 도구는 운영체제 파일시스템 할당량을 설정하지 않으며 쓰기 전 예산 검사와 임시 저장 공간 예약을 수행합니다.

Windows에서는 Python 프로세스의 committed memory를 Job Object로 제한하고 RSS도 주기적으로 검사합니다. 하드 한도 도달로 프로세스가 종료돼도 마지막 확정 체크포인트로 재개합니다. 매우 작은 한도나 큰 BatchSize는 실행 도중 실패할 수 있습니다. 검증에서는 BatchSize 32, MaxMemoryGB 2를 사용했습니다.

```text
DataDirectory/
  ledger.sqlite3                 # 원본 위치·분할·중복·샤드·체크포인트 확정 기록
  fixed/validation.jsonl         # 고정 validation
  fixed/test.jsonl               # 고정 test
  shards/shard-00000001.jsonl    # 미완료 샤드와 마지막 완료 샤드 등
  source-spool/                  # 디코드 검증용 일시 파일
  trainer.lock                  # 같은 실행의 동시 쓰기 방지
CheckpointDirectory/
  checkpoint-owner.json         # 이 ledger와 체크포인트 폴더의 연결
  checkpoints/bundle-<uuid>.pt
  checkpoints/bundle-<uuid>.mirror.pt
  checkpoints/bundle-<uuid>.zero
  checkpoints/bundle-<uuid>.manifest.json
  logs/training.csv
  logs/training.jsonl
  status.json
  resource-policy.json
  training-report.json
```

콘솔에는 전체 진행률, 현재 샤드, 완료 포지션, 학습 단계, Policy/WDL loss, validation loss, 처리 속도, ETA, RSS, 디스크 사용량과 최신 체크포인트가 표시됩니다. 현재 샤드의 Epoch 진행을 환산하므로 처리 속도는 원본 JSONL 파싱 속도와 다릅니다. 표시와 로그는 확정 저장 주기에 갱신됩니다.

검증·저장에 성공한 샤드만 삭제 대상이 됩니다. 마지막 완료 샤드는 이전 체크포인트 복구를 위해 보관하며, Replay는 체크포인트 자체에 포함합니다. `-KeepShards`는 샤드 보관을 늘리지만 모든 과거 체크포인트 보관 옵션은 아닙니다. 파일이 손상됐거나 필요한 복구 데이터가 없으면 삭제하지 않고 중단합니다.

## 8. 체크포인트 검사와 복구 도구

다음 명령은 학습하지 않고 현재 체크포인트와 미러를 검사합니다.

```powershell
.\.venv\Scripts\python.exe -m shardtrain.recover `
  --data-directory .\data\demo100 --checkpoint-directory .\runs\demo100
```

이전 정상 저장으로 ledger head를 되돌려야 할 때는 다음 명령을 사용합니다. 학습 중인 프로세스와 동시에 실행하지 마세요.

```powershell
.\.venv\Scripts\python.exe -m shardtrain.recover `
  --data-directory .\data\demo100 --checkpoint-directory .\runs\demo100 --repair
```

`-Resume`도 동일한 체크포인트 검증·fallback을 수행하므로 보통은 별도의 repair 없이 재개할 수 있습니다. 복구는 검증 가능한 보관 체크포인트와 필요한 샤드가 남은 범위에서 가능합니다. 모든 사본 또는 SQLite ledger가 손상된 경우까지 미러가 대신 복원해 주는 구조는 아닙니다.

## 9. 후보 모델 평가와 챔피언 보존

학습 완료 모델은 항상 **후보**로 저장됩니다. 학습 loss 감소만으로 승격하지 않으며 자동 챔피언 덮어쓰기 기능이 없습니다. 평가에는 같은 고정 validation/test의 Policy·WDL loss, 시작 위치 및 20개 합법 자식의 가치 일관성, 별도 실제 결과의 보정 오차, 공식 퍼즐, 같은 시간·탐색 설정의 대국이 포함됩니다.

이번에 만든 후보는 `candidates/positions-100.pt`, `positions-1000.pt`, `positions-10000.pt`와 같은 이름의 `.zero`로도 복사해 제공합니다. `.zero`는 호환 엔진에서 후보를 시험하는 용도이고, PT는 새 실험의 `-InitialCheckpoint`로 사용할 수 있습니다. 이 경우 새 Optimizer로 시작하며 과거 실행의 처리 위치를 이어가는 `-Resume`과 구분합니다. [후보 색인](candidates/index.json)에 해시와 승격 보류 근거가 있습니다.

일반 평가 명령은 `--engine`을 반드시 지정해야 합니다. 아래 경로는 이번 작업에서 검증한 기존 메모리 개선 엔진의 위치이며 실행파일 자체는 수정하지 않았습니다. 배포 ZIP에 해당 엔진이 없으면 보유한 호환 CHROMATIC 메모리 개선 엔진의 경로를 지정하세요.

```powershell
.\.venv\Scripts\python.exe .\tools\evaluate_candidate.py `
  --data-directory .\data\demo100 --checkpoint-directory .\runs\demo100 `
  --engine ..\project-chromatic-memory\dist\chromatic-memory.exe `
  --champion-pt .\champion\model.pt --champion-zero .\champion\default.zero `
  --fixtures-directory .\fixtures --pairs 2 --milliseconds 10 --max-plies 128
```

평가는 CPU1 스레드, 동일 엔진·탐색 옵션·고정 시간으로 모델만 바꿉니다. 기본은 2쌍, 수당 10ms, 최대 128플라이로 제한된 검사입니다. 미완료 대국은 `*`로 남기며 무승부로 계산하지 않습니다. 대국 수와 시간은 실제 실행량을 늘리므로 필요에 맞게 사용자가 선택하세요.

현재 승격 판단에는 각 고정 세트 1,000개 이상, 원본 프레임 무결성, 독립 보정과 누수 검사, 주요 지표 비악화, 전술·대국 엔진 오류 없음, 완료 대국 64개 이상과 점수의 95% 하한이 0.5 초과인 조건 등이 있습니다. 모두 만족해도 수동 검토 가능 여부만 보고합니다. 기존 챔피언의 과거 학습 이력을 역으로 감사했다고 주장하지 않습니다. [평가 상세](docs/EVALUATION.md)를 참고하세요.

## 10. 자동 테스트와 검증 자료

일반 단위 테스트는 네트워크 다운로드 없이 실행할 수 있습니다.

최종 검사에서 일반 테스트 **58개와 20개 subtest**, 별도 실제 프로세스 테스트 **6개**가 모두 통과했습니다. 스트리밍 테스트는 로컬 HTTP 서버로 재연결을 모사하며 대용량 원격 다운로드는 수행하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

실제 프로세스 종료·재개 통합 테스트는 여러 번의 100개 포지션 학습을 수행하므로 명시적으로 켭니다. 공식 로컬 fixture만 사용합니다. 테스트용 실행 폴더와 보고서를 새로 생성하므로 일반 학습 실행의 위치는 사용하지 않습니다.

```powershell
$env:CHROMATIC_RUN_PIPELINE_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_pipeline.py -v
Remove-Item Env:CHROMATIC_RUN_PIPELINE_TESTS
```

세 강제 종료 지점과 미러·이전 체크포인트 복구를 기준 실행과 비교해 모델, Optimizer, Scheduler, Python/NumPy/PyTorch RNG, 배치 순열, Replay, 모델 내보내기가 일치하는지 확인합니다. 단위 테스트에는 FEN·캐슬링·승격·평가 관점·라벨 가중치·스트림 손상·분할·샤드 보존과 경로 안전성 검사가 포함됩니다.

| 자료 | 내용 |
|---|---|
| [실험 보고서](reports/EXPERIMENTS.md) | 100·1,000·10,000개 실제 실행, 후보 비교와 해석 |
| [통합 복구 결과](reports/pipeline-tests.json) | 실제 프로세스 종료와 정확한 재개 결과 |
| [추론 일치 결과](reports/inference-parity.json) | 기존 네이티브 SIMD 추론과 PyTorch의 비교 |
| [독립 CP 보정 결과](reports/calibration-report.json) | 게임 단위 분할·보정·holdout 지표 |
| [원본 프레임 검증](reports/official-frame-verification.json) | 소비한 공식 평가 프레임 무결성 |
| [검증 그래프](reports/graphs/validation-and-safety.png) | 고정 검증 및 안전성 결과 |
| [시작 위치 그래프](reports/graphs/start-consistency.png) | 시작 위치와 자식의 가치 일관성 |

단계별 실제 실행의 데이터·체크포인트·원시 로그는 같은 상위 폴더의 [chromatic-shard-validation](../chromatic-shard-validation/)에 있습니다. 검증 당시 경로가 기록된 실행 자료이므로 다른 폴더로 옮긴 사본의 재개는 지원하지 않습니다. 이번 표본에서 소프트웨어 동작과 복구를 확인했으며, 학습량 증가나 작은 대국 결과만으로 Elo 향상을 입증한 것은 아닙니다.

새 학습 도구의 설계는 [ARCHITECTURE.md](docs/ARCHITECTURE.md), 데이터 처리 계약은 [STREAMING.md](docs/STREAMING.md), 모델 라벨 근거는 [LABELS_AND_CALIBRATION.md](docs/LABELS_AND_CALIBRATION.md)에 있습니다. 외부 구성요소와 라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md), [LICENSE](LICENSE)를 확인하세요.
