# 기존 CPU 샤드 실행을 원본 보존하며 가져오기

`tools/import_legacy.py`는 기존 CPU 샤드 학습기의 실행을 **새 디렉터리에 복사**한다. 원본 ledger에 ARC 설정을 쓰거나 원본 체크포인트를 변환하지 않는다. 모델 가중치뿐 아니라 Optimizer·Scheduler·Python/NumPy/Torch RNG·로컬 샘플러 generator·Replay·샤드 순열·에폭·offset·전역 step을 보존하는 경로다.

`-InitialCheckpoint`로 새 학습을 시작하는 기능과 구분한다. InitialCheckpoint는 모델 구조·가중치를 읽고 새 Optimizer를 만든다. Legacy import는 이미 진행한 학습 상태를 보존하여 CPU FP32의 같은 학습 계약으로 이어간다. 기존 실행을 ARC에서 직접 `-Resume`하여 원본을 수정하는 경로는 제공하지 않는다.

## 실행 전 조건

- 원본은 기존 CHROMATIC Lichess Shard Trainer `trainer_version=1` 실행이어야 한다. 임의의 PT 파일만 가진 경우 전체 샤드 상태를 재구성할 수 없다.
- 원래부터 정상 종료된 깨끗한 SQLite snapshot을 사용한다. `ledger.sqlite3-wal`이 비어 있지 않으면 가져오기를 거부한다. 도구가 원본에 `wal_checkpoint`, 설정 변경, 잠금 파일 생성 등을 수행하지 않는다.
- 원본의 고정 validation/test, 보존된 필요한 샤드, 정상 체크포인트 또는 미러가 남아 있어야 한다.
- 원래 입력 source·calibration·InitialCheckpoint·오프닝 목록 경로를 유지해야 한다. 입력 경로까지 이동하면 소스 cursor 정체성과 설정 서명이 달라진다.
- 목적지 최상위 폴더는 새 폴더 또는 빈 폴더여야 한다. 원본 data/checkpoint 폴더와 서로 포함 관계일 수 없다. 부분 실패 폴더를 덮어써 재사용하지 않는다.

## 명령

README가 있는 ARC 폴더에서 실행한다. 다음은 현재 작업 공간의 보존된 완료 100개 실행을 예로 든 경로다. 다른 실행에서는 source data/checkpoint 두 경로를 함께 바꾼다.

```powershell
$Python = 'C:\Users\fly_t\chromatic-xpu\Scripts\python.exe'
& $Python .\tools\import_legacy.py `
  --source-data-directory ..\chromatic-shard-validation\positions-100\data `
  --source-checkpoint-directory ..\chromatic-shard-validation\positions-100\candidate `
  --destination-root .\runs\legacy100-copy --max-copy-gb 4 --dry-run
```

Dry run은 원본을 읽어 일관성·해시·복구 가능 여부와 복사량을 검사하고 목적지를 만들지 않는다. 신뢰할 수 있는 로컬 PT를 읽으므로 이 명령은 장치 없는 설정 DryRun과 달리 PyTorch를 import한다. 실제 복사는 `--dry-run`을 뺀 같은 명령이다.

```powershell
& $Python .\tools\import_legacy.py `
  --source-data-directory ..\chromatic-shard-validation\positions-100\data `
  --source-checkpoint-directory ..\chromatic-shard-validation\positions-100\candidate `
  --destination-root .\runs\legacy100-copy --max-copy-gb 4

.\chromatic-train.ps1 -Python $Python -Resume `
  -DataDirectory .\runs\legacy100-copy\data
```

첫 Resume는 `runs/legacy100-copy/candidate`를 자동으로 찾아 검증된 기존 상태를 읽고, 새 복사본의 CPU 실행 설정으로 한 번만 이행한다. 원래 실행이 이미 완료됐다면 학습 업데이트를 더하지 않는다. 새 ARC 모델 head/서명만 새 복사본에 기록한다. 원래 실행이 진행 중이던 상태라면 복사된 순열·offset·Optimizer 상태에서 이어간다.

## 무엇을 복사하고 바꾸는가

원본 SQLite는 `mode=ro&immutable=1`로 열고 읽기 트랜잭션에서 SQLite backup API로 새 ledger를 만든다. 비어 있지 않은 WAL을 무시한 채 immutable로 읽지 않도록 먼저 거부하고 마지막에도 다시 확인한다. 원본을 동시에 학습·정리하는 상태에서 사용하는 도구가 아니다.

복사 범위는 ledger, 고정 holdout, ledger가 가리키는 보존된 소유 샤드, 등록된 체크포인트 bundle이다. 원래 디렉터리의 임의 사용자 파일·일시 spool·잠금 파일을 복사하거나 지우지 않는다. PT 본체·미러·`.zero`는 **바이트 그대로 복사**한다. 체크포인트 내부의 기존 settings/signature·모델·학습 상태도 그대로다.

새 ledger와 새 manifest에는 다음의 목적지 경로만 다시 기록한다.

- 고정 holdout의 새 파일 경로
- 소유 샤드의 새 파일 경로
- bundle 파일·head·샤드 완료 receipt·이벤트의 새 체크포인트 경로
- 새 data/checkpoint 디렉터리의 실행 설정과 새 owner token

이미 보존 기간이 지나 삭제된 이전 bundle을 가리키는 정상적인 과거 완료 receipt·로그도 소유 파일명 규칙을 확인한 뒤 새 경로로 매핑한다. 해당 과거 파일을 만들어 내거나 오래된 완료 샤드를 다시 학습하지 않는다. 현재 head가 손상됐으면 검증 가능한 미러/이전 상태와 필요한 샤드가 남은 범위를 검사하며, 원본 자체를 repair하지 않는다.

새 ledger의 `legacy_import` marker는 `pending_arc_migration`, `copy_complete=true`로 기록한다. 이 marker는 복사·경로 변경·새 DB·고정 세트·체크포인트 검증과 원본의 전후 SHA 일치를 확인한 **뒤에만** 확정한다. 기존 settings와 signature는 별도로 보존하고 최초 ARC Resume가 제한된 버전 이행을 승인하는 근거로 사용한다.

## 첫 Resume의 제한

초기 이행은 CPU, FP32, 1스레드, 기존 배치·Epochs·ShardSize·Positions·학습률·Seed·스케줄러·Replay 설정으로만 허용한다. ARC 측 추가 실행 조건은 `prefetch=0`, `cache=auto`, `transfer=copy`, `optimizer=default`, `gradient_accumulation=1`, `fallback=error`다. 기존 데이터/라벨/초기 가중치 해시와 원본 경로도 일치해야 한다.

기존 v1의 검증 의미는 full이므로 최초 이행도 full이다. 새 실행의 기본 selected를 복사된 과거 실행에 자동 적용하지 않는다. 실제 보존된 완료 100개 실행을 이 경로로 가져와 ARC에서 재개했을 때 step 12, 모델/Optimizer/Scheduler/RNG/Replay/진행 등 10개 의미 상태와 `.zero`가 같았고 신규 학습 업데이트는 0회였다. [실제 import 검증](../reports/legacy-import-validation.json)에 원본 읽기 전용과 새 head 기록 근거가 있다.

학습 목표를 늘리거나 배치·정밀도·장치를 바꿔 기존 진행을 재해석하지 않는다. XPU 또는 다른 설정의 새 실험을 원하면 새로 저장한 CPU-portable PT를 다른 새 실행의 InitialCheckpoint로 사용한다. 이 경우 Optimizer와 처리 위치를 이어가는 import와 구분한다.

NumPy RNG는 새 seed로 재생성하지 않고 기존 MT19937 상태 배열·커서를 저장한 그대로 읽어 `set_state`로 복구한다. Python/Torch와 로컬 sampler 상태도 보존한다. 검증한 Python/PyTorch/NumPy 조합을 유지하는 것이 좋으며 다른 버전에서 읽기 실패하면 원본을 변환하지 않고 중단한다. 서로 다른 장치의 학습 결과가 비트 단위로 같다는 주장은 하지 않는다.

## 실패와 증거

검증 실패나 복사 중단은 목적지에 `import-incomplete.json`과 가능한 경우 `legacy_import_failed` 기록을 남긴다. 이 실패 복사본은 정상 Resume와 재-import 대상이 아니다. 원본을 보존한 채 새 빈 목적지에서 다시 시작한다. 복사 도구가 재귀적으로 실패 폴더를 삭제하지 않는다.

성공하면 목적지의 `import-report.json`에 원본 DB/head SHA, 원본·새 경로, step/commit, 미러/rollback 여부, 파일별 복사 SHA와 원본 전후 불변 확인을 기록한다. 새 PT·미러·`.zero`의 해시는 원본과 같고 manifest의 경로는 새 소유 폴더를 가리킨다.

회귀 시험은 실제 새 Python 프로세스로 원본 전체 파일 SHA와 부가 파일 생성 여부를 비교하고, 모델·Optimizer·Scheduler·generator·Replay·진행 상태 보존, 비어 있지 않은 WAL 거부, 고정 세트 손상, 복사 예산, 부분 실패 재사용 거부, 미러 복구, 최종 검증 실패 시 marker 미발행을 검사한다. 실제 이전 실행을 가져와 재개한 결과와 버전 이행 증거는 [PERFORMANCE.md](../PERFORMANCE.md)의 원시 보고서 링크를 따른다.
