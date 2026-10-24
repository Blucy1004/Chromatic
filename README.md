# CHROMATIC

독립 C++20 PUCT-MCTS 체스 엔진과 PyTorch Policy/WDL 샤드 학습 도구입니다.
엔진 버전, 챔피언 모델, 학습 후보와 실험 기록을 각각 보존합니다.
현재 저장소는 2026-10-24에 검증한 Windows x64 개발 결과를 모은 초기 공개본입니다.

## 구성

| 폴더 | 내용 |
| --- | --- |
| [project-chromatic](project-chromatic/README.md) | 보존된 CHROMATIC 1.0 소스·Windows 실행 파일·기본 모델 |
| [project-chromatic-depth](project-chromatic-depth/README.md) | 선택적으로 켜는 MCTS 깊이 계측 버전 |
| [project-chromatic-memory](project-chromatic-memory/README.md) | 메모리 포화 후 얕은 평가 반복을 고친 별도 탐색 버전 |
| [chromatic-shard-trainer](chromatic-shard-trainer/README.md) | 보존된 CPU Lichess 회전형 샤드 학습기 |
| [chromatic-arc-trainer](chromatic-arc-trainer/README.md) | CPU / Intel Arc XPU 학습·프로파일·자동 장치 선택·복구 도구 |
| chromatic-shard-validation | 기존 100 / 1,000 / 10,000개 실험의 보고서·고정 데이터 |
| chromatic-cause-analysis / chromatic-depth-analysis / chromatic-memory-analysis | 초기 수 선택 원인·깊이·메모리 회수 비교의 원시 결과와 분석 |

Muyeong은 포함하지 않습니다. 학습 후보를 기존 챔피언에 자동 적용하는 기능은 없습니다.

## Windows UCI 실행

CHROMATIC 메모리 개선 버전을 체스 GUI의 UCI 엔진으로 등록합니다.
실행 파일과 `default.zero`를 같은 폴더에 둡니다.

```powershell
.\project-chromatic-memory\dist\chromatic-memory.exe
```

소스 빌드는 해당 엔진 폴더에서 `build.cmd`를 실행합니다. MSVC C++20, CMake와 Ninja가 필요합니다.
보존된 1.0을 실행하려면 `project-chromatic/dist/chromatic.exe`를 사용합니다.

## 학습 실행

검증 환경은 Python 3.11.9 / PyTorch 2.14.1+xpu / Intel Arc 140T입니다.
전용 가상환경을 자동 변경하거나 패키지를 자동 설치하지 않습니다.
다음은 포함된 공식 CC0 로컬 표본을 사용하는 CPU 4스레드 실행 예입니다.

```powershell
cd .\chromatic-arc-trainer
.\chromatic-train.ps1 -Positions 10000 -ShardSize 1000 -Epochs 3 `
  -Source .\fixtures\official-evaluations.jsonl.zst `
  -DataDirectory .\data\cpu4-10k -CheckpointDirectory .\runs\cpu4-10k `
  -Device cpu -CpuThreads 4 -BatchSize 32 -Precision fp32 `
  -ValidationMode selected -ReplaySize 256 -Seed 2026 `
  -MaxMemoryGB 4 -MaxDiskGB 1 -MaxDownloadMB 16
```

`-DryRun`을 붙이면 학습·다운로드 없이 설정만 검사합니다.
XPU는 `-Device xpu`, 중단된 동일 실행의 복구는 `-Resume`으로 선택합니다.
설치, 제한, 전체 옵션은 [ARC README](chromatic-arc-trainer/README.md)를 확인하세요.

## 실제 검증 결과

동일 Python 환경·동일 10,000개 데이터·3에폭·배치 32에서 기존 CPU 학습기는
2회 중앙값 93.77초, 개선 CPU 4스레드는 3회 중앙값 58.84초(1.59배),
Arc XPU FP32는 3회 중앙값 71.28초였습니다.
기존 학습기 관측 범위는 82.25~105.29초이며 환경·부하 차이를 완전히 제거한 실험은 아닙니다.
작은 MLP에서는 CPU가 가장 빨랐고, XPU FP32/BF16 학습도 실제로 동작했습니다.

- 회귀 검사 151개 + 하위 검사 24개 통과(별도 실행 검사 6개는 일반 suite에서 제외).
- CPU/XPU의 강제 종료·미러·이전 정상본 복구 등 실제 시나리오 20개 통과.
- 후보 3종 × 입력 10개, 독립 C++ 추론 일치 30/30 통과.
- 공식 60,000행의 full/selected 정규화 의미 비교 불일치 0건.
- 기존 엔진·모델·학습기·이전 실험 파일 515개 해시 보존.

[PERFORMANCE와 그래프](chromatic-arc-trainer/PERFORMANCE.md),
[정확성·복구 계약](chromatic-arc-trainer/docs/VALIDATION.md),
[원시 CSV](chromatic-arc-trainer/reports/benchmarks/all-runs.csv),
[원시 JSONL](chromatic-arc-trainer/reports/benchmarks/all-runs.jsonl)을 함께 제공합니다.
완료되지 않은 메모리 한도 실패·시간 제한 중단과 진단용 이전 결과도 범위를 표시해 보존했습니다.
이 속도·수치 결과는 Elo 향상이나 챔피언 승격의 근거를 대신하지 않습니다.

## 보존과 재현 범위

`repository-manifest.json`은 이 초기 공개본에 복사한 파일의 SHA-256을 기록합니다.
기존 결과의 원본 내용과 해시는 바꾸지 않았으므로 일부 과거 로그에는 실행 당시의 Windows 절대 경로가 남습니다.
전체 파이프라인 추천은 모델·환경·데이터·실행 조건과 근거 파일 해시가 맞을 때만 재사용하며,
새 경로로 복사한 뒤 과거 근거를 확인할 수 없으면 측정 기반 선택으로 돌아갑니다.
가상환경, 빌드 중간 파일과 재생성 가능한 캐시는 포함하지 않습니다.
새 ARC 실행의 SQLite 상태와 중복 체크포인트 묶음은 제외했습니다.
예외로 `chromatic-shard-validation`의 기존 100/1,000/10,000개 실행 상태 3종은
이전 CPU 형식의 import·복구 증거를 보존하기 위해 원래 DB와 체크포인트를 그대로 포함합니다.
완전한 로컬 실행 상태를 옮긴 Resume와 후보 PT로 새 실행을 시작하는 것은 다릅니다.

## 라이선스

프로그램은 [GPL-3.0](LICENSE)입니다. 데이터는 공식 Lichess CC0 출처이며,
각 fixture의 `.source.json`과 각 구성요소의 `THIRD_PARTY.md`에 세부 이력을 보존합니다.
기존 챔피언의 역사적 학습 이력 전체를 CC0로 새로 보증한다는 뜻은 아닙니다.
