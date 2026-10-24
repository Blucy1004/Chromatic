# ARC 데이터 파이프라인과 계측 계약

이 문서는 별도 ARC 사본의 실제 구현을 설명한다. 이전 학습기의 데이터 품질·고정 분할·원자적 체크포인트·샤드 회전 의미를 유지하면서 반복되는 FEN/합법수 인코딩과 데이터 전송을 줄인다. 기존 엔진·Muyeong·이전 실험 결과를 수정하지 않는다.

## 1. 원본에서 정규화 샤드까지

`RecordStream`은 공식 CC0 원본을 작은 청크로 순차 읽고 Zstandard 프레임을 제한된 일시 spool에 해제한다. 전체 프레임을 RAM에 올리지 않는다. 프레임 끝의 검증이 끝난 후에만 행을 노출한다. 입력 byte 범위, 누적 해제량, 프레임 spool, 선언 window, 한 줄 길이를 각각 제한한다. 현재 단일 프레임 spool 상한은 128MiB, 한 줄은 1MiB, 선언 window는 128MiB다. 매우 큰 프레임을 우회해 부분 학습하는 것이 기본 동작은 아니다.

소스 계층은 URL·라이선스·ETag/버전·길이·로컬 해시와 행 접두부 SHA를 기록한다. 살아 있는 디코더의 HTTP 재연결은 정확한 후속 Range를 요청하고, 서버가 Range를 무시하면 이미 읽은 압축 접두부를 버린 뒤 디코더에 중복 입력하지 않는다. 새 프로세스는 압축 byte 0부터 재생해 저장된 행 prefix 해시를 맞춘다. `compressed_offset_diagnostic`는 디코더 재시작 지점이 아니다.

체크섬 없는 프레임은 경계·디코딩 완료와 체크섬 검증을 구분해 `complete_frames_without_full_checksum`으로 표시한다. 명시적인 부분 프레임 조사 기능은 `partial_frame_checksum_unavailable`로 표시하고 기본 학습에서는 사용하지 않는다. 단순히 압축 해제가 됐다고 원본의 전체 무결성이 증명되는 것은 아니다.

JSONL 파싱 다음에 FEN, 교사 depth/knodes, PV 전체 수순의 합법성, 백→현재 차례 평가 관점, cp/mate 라벨, 정확한 identity와 family를 검증한다. 동일 identity는 SQLite에서 한 번만 수용한다. 기물·폰 구조 집중 상한과 초반/중반/종반 균형, family를 넘지 않는 학습/validation/test 분할을 유지한다. 이 단계에서 바뀌는 행 결정과 소스 cursor는 같은 SQLite 트랜잭션으로 확정한다.

현재 `ValidationMode=selected`는 평가 metadata의 우선순위를 먼저 정하고 실제 사용할 후보부터 PV 전체 수순을 검증한다. 선택 후보가 잘못됐으면 다음 후보를 검사한다. 이미 유효한 선택이 결정된 뒤 사용하지 않는 낮은 평가까지 반복 검사하지 않는다. `full`은 기존 모든 평가 검사 및 rejection-audit 범위를 유지한다. 공식 60,000행의 선택된 평가·정규화 정책/WDL·가중치·FEN/history·identity·provenance가 0 mismatch였고, selected CPU/XPU의 같은 데이터 학습과 강제 종료/Resume 결과도 확인한 뒤 selected를 새 실행 기본값으로 채택했다. 사용하지 않은 분석의 세부 감사 목록은 명시적으로 비교 범위가 다르다.

공식 평가 데이터의 4필드 FEN에는 실제 게임 ply·halfmove·반복 이력이 없다. 초반 분류의 기물 수 대용 지표와 실제 0~12플라이 오프닝 앵커를 구분한다. 앵커는 약한 Policy 모방이며 Value 가중치는 0이다. CP→WDL은 별도 실제 게임 결과로 보정한 약한 라벨이고 단일 PV는 완전한 정책 분포가 아니다. 보정·전술 family와 고정 holdout은 Replay나 새 학습 샤드에 들어가지 않는다.

## 2. 검증된 compact binary

`fastdata.EncodedCache`는 보존된 인코더를 한 번 호출해 검증된 정규화 행을 이진 NPY에 쓴다. 매 에폭마다 FEN을 복구하고 합법수를 다시 만드는 작업을 대신한다. 모델의 데이터 의미나 confidence 가중치를 바꾸지 않는다.

|이진 필드|형식|의미|
|---|---|---|
|`x`|1,280 × FP32|기존 20개 평면 인코딩|
|`legal`|584 × uint8|4,672개 합법수 mask를 bit-pack|
|`policy_actions`|256 × int16|관측된 sparse Policy action, 빈칸 −1|
|`policy_prob`|256 × FP32|해당 정책 확률, 기존 FP32 합산 결과 유지|
|`wdl`|3 × FP32|현재 차례 W/D/L|
|`pw`, `vw`|각 FP32|독립 Policy/Value confidence 가중치|

현재 row는 7,260바이트다. 4,672개 dense Policy와 bool mask는 **요청한 미니배치에만** 확장한다. 정규화 source의 Policy support가 안전 스키마 256개를 넘으면 조용히 자르지 않고 거부한다. 현재 공식 eval PV 제한은 16개이며 오프닝 앵커도 이 상한 이내다.

캐시 계약에는 schema, 기존 `encoding.py` SHA, 입출력 크기, 행 순서, identity, **정규화 행 전체의 fingerprint**, 원본 SHA를 전달한 경우 해당 값이 포함된다. 같은 FEN의 라벨·confidence가 달라져도 이전 캐시를 재사용하지 않는다. NPY 자체 SHA와 dtype/shape도 검사한다. 이 검사는 우연한 stale/corrupt 캐시 재사용을 방지하며 원본 출처 검증을 대신하지 않는다.

새 binary는 고유 temp 파일에 제한된 청크로 생성하고 flush/fsync·원자적 replace 뒤 manifest를 확정한다. 중단된 temp를 정상 캐시로 읽지 않는다. 손상된 정상 파일은 자동 삭제·재생성으로 숨기지 않고 거부해 검사할 수 있게 남긴다. 캐시는 파생 자료이며 학습 성공 여부의 권위 상태는 ledger와 검증된 체크포인트다.

## 3. 캐시 모드와 메모리 범위

|모드|현재 동작|
|---|---|
|`none`|기존 참조 인코더로 각 요청 배치를 생성|
|`memory`|검증된 compact NPY를 CPU RAM에 복사, 예산 초과 시 오류|
|`mmap`|검증된 NPY를 read-only memory mapping으로 열고 필요한 행만 요청|
|`auto`|남은 캐시 예산 안에 들어가는 작은 세트는 memory, 나머지는 mmap|
|`device`|명시적 XPU 현재 샤드+Replay dense 상주 실험; 한도 초과 시 오류|

일반 CPU 캐시의 전체 resident 예산은 `min(MaxMemoryGB × 15%, 256MiB)`다. 현재 샤드와 Replay, 고정 validation/test를 함께 고려한다. 작은 replay miss용 hot cache도 이 예산 안에서 제한한다. GPU 상주 모드는 dense 확장량을 별도로 계산해 `min(MaxMemoryGB × 15%, 256MiB)`를 넘으면 `auto` 또는 `mmap`을 사용하도록 중단한다. 이 한도는 전체 프로세스 한도와 별개로 더 작은 내부 한도다.

현재 학습 루프는 **현재 샤드의 정규화 JSON 행과 제한된 Replay**를 한 번 읽고, 이후 에폭에서는 검증된 binary를 인덱스로 읽는다. 전체 데이터베이스를 RAM에 올리는 구조는 아니다. 큰 샤드의 binary feature는 mmap으로 읽을 수 있지만 현재 샤드의 정규화 메타데이터까지 완전한 무상주 iterator라고 주장하지 않는다. ShardSize 최대 100,000과 MaxMemoryGB를 함께 설정해야 하며 한도 안에 들지 않으면 중단된다.

`BatchBuffer`는 dense CPU 배열을 재사용하는 별도 실험용 API다. 이전 배치를 읽는 비동기 전송이 끝나기 전에 같은 버퍼를 덮어쓰면 안 된다. 기본 prefetch는 각 배치가 자신의 배열을 소유해 이 경쟁을 피한다. XPU `preallocated` 전송도 한 배치 버퍼의 수명을 명시적으로 관리하며 기본값으로 활성화하지 않는다.

마지막 완료 샤드는 체크포인트 rollback을 위해 유지한다. 기본 모드는 학습 완료 후 해당 파생 binary를 닫고 삭제하며, 정상 완료하지 않은 원본 샤드의 보존 정책은 유지한다. `KeepShards`는 원본 샤드와 이진 캐시 보관량을 늘리므로 디스크 예산을 함께 검토한다.

## 4. 두 생산자·소비자 경로

`Prefetch=0`은 순차 경로다. 1~16은 두 종류의 제한 큐를 활성화한다. 큐 길이를 늘려도 원본 검증을 생략하거나 학습 순서를 바꾸지 않는다. 데이터가 적은 경우 queue·thread 오버헤드로 더 느릴 수 있으므로 효과 확인 없이 기본값으로 채택하지 않는다.

**원본 행 큐**는 단일 worker가 RecordStream을 소유한다. 각 항목은 `(line, text, cursor_snapshot, stats_snapshot)`이다. worker가 원본을 몇 줄 앞서 읽어도 소비자가 마지막으로 받은 cursor만 SQLite에 확정한다. 미래의 `stream.cursor`를 현재 트랜잭션에 기록하지 않는다. 정상 끝은 FIFO로 전달하고 실패는 해당 순서에서 소비자에 다시 발생시킨다. 압축 해제 작업은 학습과 겹칠 수 있지만 완성된 프레임 spool·입력 예산·큐 길이 제한을 유지한다.

**배치 큐**는 이미 고정된 샤드 순열과 Replay 선택 계획을 CPU 텐서로 만든다. 에폭의 권위 generator를 복사한 로컬 CPU generator에서 앞선 Replay 인덱스를 계산하고, 각 계획에 `rng_after`를 담는다. 메인 학습기는 **실제로 Optimizer 단계를 마친 배치의 rng_after만** 권위 generator에 반영한다. worker의 읽기 선행이 checkpoint RNG나 진행 offset을 앞당기지 않는다.

이 경로에는 worker의 SQLite 연결이나 장치 학습 RNG가 없다. source worker는 bytes/프레임/행만 만들고 정규화·SQLite 결정은 소비 스레드에서 한다. 배치 worker는 CPU 인덱싱·확장만 수행하고 모델 forward/backward/Optimizer는 메인 스레드에서 한다. 데이터 순서·epoch offset·Replay 선택을 저장해 같은 장치·같은 설정의 중단/재개를 비교한다.

Queue 최대 길이는 설정값이고 worker가 만드는 한 항목과 소비 중인 항목이 추가로 있을 수 있다. 조기 종료 시 stop event를 설정하고 producer를 join한 뒤 generator와 원본을 닫는다. Guard의 디스크 예약량·스캔·RSS 검사에는 재진입 잠금을 사용해 source worker와 체크포인트 저장 스레드가 동시에 상태를 덮어쓰지 않도록 한다. 프로세스 강제 종료가 남긴 소유 temp는 다음 실행 잠금을 얻은 뒤 제한된 이름/경로에서만 정리한다.

## 5. 모델 계산과 저장 경계

모델과 배치는 선택한 CPU/XPU에 배치한다. BF16에서는 autocast forward를 사용하되 softmax·교차 엔트로피는 FP32로 계산한다. Validation은 FP32다. Gradient clipping, 모델·gradient·Optimizer finite 검사를 유지한다. Optimizer 이전 BF16 비정상 배치는 partial gradient를 버리고 전체 유효 배치를 FP32로 다시 계산하며 이 선택을 기록한다.

BF16은 첫 업데이트와 매 32번째 업데이트에서 같은 배치의 FP32 reference loss를 검사한다. BF16 악화가 `max(0.02, abs(FP32 loss) × 0.02)`를 넘으면 Optimizer 전에 전체 유효 배치를 FP32로 재시도한다. reference 비교 결과와 fallback은 실행 보고서에 남으며, 이 한 배치 검사가 독립 holdout 또는 기력 검증을 대체하지 않는다.

체크포인트는 장치 실행을 동기화한 뒤 모든 모델·Optimizer tensor를 독립적인 CPU storage로 복사한다. XPU RNG도 CPU uint8 tensor로 저장한다. 별도 CPU 모델 사본으로 기존 `.zero` 내보내기와 파일 검증을 수행해 C++ 추론 형식을 바꾸지 않는다. GPU allocator에만 남은 임시 storage에 체크포인트 파일이 의존하지 않는다.

동일 ledger의 장치·정밀도·배치·Optimizer 설정 변경은 Resume signature가 거부한다. 다른 장치에서 PT를 읽을 수 있다는 것과, 장치를 바꾼 재개가 비트 단위로 같다는 것은 다른 조건이다. 장치 이동의 새 실험은 InitialCheckpoint에서 모델 가중치를 읽고 새 Optimizer로 시작한다.

ValidationMode가 없던 이전 ARC 상태는 Resume에서 full로 복원하고 full의 이전 서명 형태를 유지한다. 기존 실행을 새 기본 selected로 조용히 바꾸지 않는다. 기존 v1 CPU 실행 전체 상태를 가져오는 경우도 원본 읽기 전용 복사와 제한된 최초 CPU/full 이행을 사용한다. 실제 완료된 100개 실행의 import에서 step 12와 모델·Optimizer·Scheduler·RNG·Replay·진행 등 10개 상태가 같고 추가 학습은 0회였음을 별도 검증했다.

## 6. 단계 시간과 자원 계측

`-Profile`은 다음 stage를 선택적으로 기록한다. CPU 함수의 비활성 경로는 활성 profiler 검사 후 원래 함수를 호출하며 장치 동기화나 이벤트 파일을 만들지 않는다.

|영역|대표 stage|
|---|---|
|원본|`source_read`, `zstd_decompress`, `source_spool_write`, `source_spool_read`|
|품질|`json_parse`, `fen_validation`, `pv_validation`, `label_conversion`|
|샤드|`dedup_sampling`, `shard_serialize`, `shard_deserialize`, `data_preparation`|
|binary/배치|`binary_cache_prepare`, `binary_encode`, `data_loading`, `source_prefetch_wait`|
|장치|`host_to_device`, `device_batch_gather`, `forward`, `backward`, `gradient_clip`, `optimizer`, `finite_check`|
|검증/복구|`validation`, `checkpoint_save`, `checkpoint_housekeeping`, `checkpoint_recovery`|

각 stage는 exclusive seconds와 inclusive seconds를 함께 기록한다. 예를 들어 PV 검증이 label 변환 안에서 호출되면 parent exclusive 시간에서는 해당 child 시간을 뺀다. 다른 스레드가 동시에 실행되면 stage 합계가 wall time보다 커질 수 있다. 모든 stage를 더해 전체 시간을 재구성하거나 data_preparation 같은 inclusive 부모와 child를 중복 합산하지 않는다.

GPU forward/backward/Optimizer·전송·검증 stage는 Profile 실행에서 전후 `torch.xpu.synchronize()`로 실제 완료 시간을 잰다. CPU 데이터 worker는 GPU를 동기화하지 않는다. Profile을 끄더라도 순수 학습·전송 총 구간은 XPU 동기화된 경계로 측정한다. 상세 Profile의 추가 동기화는 처리량을 낮출 수 있으므로 같은 조건의 Profile-off 전체 실행과 구분한다.

프로세스 자원은 약 0.2초마다 CPU process/system%, RSS, I/O byte/operation, XPU allocated/reserved/peak 값을 기록한다. GPU%는 Windows PDH `GPU Engine(*)\\Utilization Percentage`에서 현재 PID에 해당하는 유효 engine의 최댓값이다. Task Manager의 프로세스 GPU 의미와 비슷한 지표이며 실행 유닛 점유율은 아니다. counter가 없거나 rate warmup/권한/드라이버로 유효 샘플이 없으면 NA와 이유를 남긴다. NA를 0으로 평균하지 않는다.

원시 샘플은 파일로 순차 쓰고 메모리에는 제한된 tail만 유지한다. summary의 평균은 해당 지표의 유효 관측값만 사용한다. process CPU%는 여러 코어를 사용하면 100%를 넘을 수 있고 I/O는 물리 드라이브 전체 트래픽이 아니다. Arc 140T처럼 공유 시스템 메모리를 쓰는 GPU에서는 allocator bytes와 process RAM이 겹칠 수 있으므로 단순 합산하지 않는다.

## 7. 저장 파일과 점검

```text
DataDirectory/
  ledger.sqlite3
  fixed/validation.jsonl
  fixed/test.jsonl
  shards/shard-00000001.jsonl
  source-spool/                 # 프레임 검증용 일시 자료
  encoded-cache/encoded-*.npy  # 검증된 파생 binary
  encoded-cache/encoded-*.json # 계약·원본·파일 SHA
CheckpointDirectory/
  checkpoints/bundle-*.pt
  checkpoints/bundle-*.mirror.pt
  checkpoints/bundle-*.zero
  checkpoints/bundle-*.manifest.json
  logs/training.csv
  logs/training.jsonl
  profile/stages.jsonl         # -Profile 시
  profile/profile.json
  resources/resources.csv
  resources/resources.jsonl
  resources/resource-summary.json
  device-recommendation.json   # auto 선택 시
  training-report.json
  status.json
```

캐시 exact-reference 비교, stale 라벨 무효화, cache SHA 손상, memory/mmap 예산, 순서·bounded prefetch·producer 오류·조기 종료, source read-ahead cursor 확정·트랜잭션 rollback은 회귀 시험으로 확인한다. 실제 강제 종료와 동일 장치 Resume는 [VALIDATION.md](VALIDATION.md)의 별도 시험으로 확인한다. 처리량 개선 채택 여부와 원시 측정 자료는 [PERFORMANCE.md](../PERFORMANCE.md)를 따른다.

이전 작업에서 수집한 공식 fixture·라이선스·원본 프레임 체크섬 증거는 같은 데이터 바이트의 출처 증거다. 상속한 소스/시험 해시를 이번 ARC 구현 해시로 바꿔 부르지 않는다. 이번 정규화·성능·수치·복구 시험에는 각각 자체 코드/입력 SHA와 raw 결과를 기록한다. [출처 기록](SOURCES.md)과 [이번 selected 감사](../reports/quality-selected/validation.json)를 구분한다.
