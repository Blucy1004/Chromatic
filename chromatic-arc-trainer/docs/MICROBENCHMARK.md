# 장치와 배치 마이크로벤치마크

**이 표는 XPU Gradient clipping의 foreach 경로를 교정한 뒤 같은 조건으로 재측정한 결과다.** 초기 실험에서 XPU 기본 foreach 경로가 전체 gradient norm을 실제보다 크게 계산해 과도하게 clip하는 결함을 발견했다. 현재 학습 및 벤치마크는 동일한 `devices.clip_gradients()`를 사용해 XPU에서 `foreach=False`를 지정한다. CPU의 기존 경로는 유지한다. 수정 전 결과와 추천은 `reports/benchmarks/pre-clip-fix/`에 진단 전용으로 보존하며 현재 추천에 사용하지 않는다. [실제 gradient norm 진단](../reports/clip-diagnostic.json)을 참고한다.

이 결과는 **이미 인코딩한 실제 고정 검증 레코드**의 미니배치 연산 실험이다. 같은 초기 챔피언 가중치로 매 반복을 다시 시작하고, 이 실험에서 갱신한 모델은 학습 후보나 챔피언으로 저장하지 않는다. 원본 holdout 파일도 수정하지 않는다. 전체 파이프라인 및 독립 체스 품질 비교는 PERFORMANCE.md의 별도 실험을 따른다.

Windows x64 / Python 3.11.9 / PyTorch 2.14.1+xpu / CHROMATIC MLP width 64, blocks 2. 각 조건은 warmup 2회 뒤 15 Optimizer 업데이트를 3회 반복하고 중앙값을 사용했다. 전체 57개 조건을 50.73초에 완료했다. 같은 mini-batch를 반복하므로 데이터 로딩·FEN 검사·샤드 교체 비용은 이 표에 포함되지 않는다.

전송과 Forward/Backward/Optimizer 구간 전후에 `torch.xpu.synchronize()`를 호출했다. 표의 samples/sec는 전송·finite 검사·gradient clipping을 포함한 해당 구간 처리량이다. FP32 master weights를 유지하고 BF16에서는 forward만 autocast하며 softmax/교차 엔트로피는 FP32로 계산한다.

|장치 / 정밀도 / CPU 스레드|Batch 32|64|128|256|512|
|---|---:|---:|---:|---:|---:|
|CPU / FP32 / 1|4,024|6,010|9,742|10,977|9,377|
|CPU / FP32 / 2|3,496|8,065|7,825|15,412|27,583|
|CPU / FP32 / 4|8,865|13,821|21,720|32,065|43,952|
|CPU / FP32 / 8|8,885|19,310|26,025|41,680|49,121|
|XPU / FP32 / 1|3,558|5,949|11,562|24,834|36,900|
|XPU / BF16 / 1|3,840|7,326|11,540|22,880|39,580|

## Batch 256의 추가 비교

|장치 / 정밀도|기본 copy|사전 할당 전송|상주 미니배치|AdamW fused|누적 2회|누적 4회|
|---|---:|---:|---:|---:|---:|---:|
|CPU / FP32|10,977|12,154|11,811|13,304|13,019|12,238|
|XPU / FP32|24,834|27,360|32,013|22,609|19,345|15,344|
|XPU / BF16|22,880|21,727|28,068|17,342|14,677|11,602|

CPU에서는 copy/preallocated/resident가 모두 같은 CPU 텐서를 반환하는 경로다. 따라서 이 세 열의 차이는 순서·온도·측정 변동이며 CPU 전송 최적화 효과의 증거가 아니다. XPU resident는 반복 미니배치를 GPU에 한 번 올려 둔 실험으로, 전체 학습 데이터를 GPU에 올리는 기본 정책을 뜻하지 않는다. 상주 준비 시간도 JSON에 따로 남겼다.

현재 모델에서는 CPU 8스레드·batch 512의 encoded-batch 처리량이 가장 높았다. XPU FP32 batch 512는 CPU 1스레드보다 빠르지만 CPU 8스레드보다 느렸다. 수정 후 재측정에서 BF16은 batch 512에서 FP32보다 약 7.3% 빨랐지만 batch 256에서는 약 7.9% 느렸다. 사전 할당 전송은 FP32 batch 256에서는 약 10.2% 이득이 있었고 BF16에서는 오히려 느려졌다. 따라서 정밀도·전송 모드에 따른 이득이 모든 조건에서 일관된 것은 아니다. Gradient accumulation 2/4는 XPU에서 처리량을 낮췄고 fused Optimizer도 FP32 기본 Optimizer보다 느렸다. 전체 파이프라인과 품질 검증 없이 이 기능들을 기본값으로 채택하지 않는다.

57개 조건 모두 finite loss·gradient·Optimizer state 검사를 통과했다. 같은 batch 크기의 CPU FP32 기준과 비교한 최종 반복 배치 loss의 상대 차이는 최대 0.233%였다. 이는 짧은 수치 안정성 검사이며 고정 검증 loss나 전술·대국의 독립 결과를 대체하지 않는다. GPU 최대 텐서 할당은 62.24MiB였다. 프로세스 RAM과 시스템 공유 메모리의 전체 사용량은 파이프라인 자원 기록을 참고한다.

## 자동 선택과 API

`hardware_signature(model_config)`는 모델 구조, 입력/Policy 크기, PyTorch 버전, CPU·운영체제, GPU 이름·메모리·Level-Zero 드라이버를 묶는다. 모델 구조나 환경이 바뀌면 기존 추천을 재사용하지 않는다. 원시 결과와 추천은 `reports/benchmarks/by-signature/<key>/`에도 별도로 보존한다.

`recommend_for_constraints(matrix_path, model_config, cpu_threads=..., batch_size=..., precision=...)`는 사용자가 정한 배치·스레드·정밀도에 맞는 실제 측정행만 비교한다. 배치를 조용히 바꾸지 않는다. 해당 조건의 자료가 없으면 `quick_recommend(initial, tensors, ...)`가 CPU/XPU의 짧은 실제 측정을 수행할 수 있다. `recommendation.json`의 최대 처리량 배치는 권고이며 독립 품질 확인 없이 기본 학습 배치로 승격하지 않는다.

`DeviceRuntime`은 명시적인 CPU/XPU 선택, autocast, synchronize, 메모리 계측 및 RNG 복원을 제공한다. 명시적인 XPU 요청 실패는 기본적으로 오류이며, `fallback="cpu"`를 선택했을 때만 CPU FP32로 전환한다. 실행 중 BF16 수치 실패 뒤에는 소유 학습 루프가 실패 배치 이전 상태로 돌아간 뒤 FP32로 재시도해야 한다. `cpu_state()`는 모든 Tensor를 독립적인 CPU 복사본으로 만들며 `.zero` 출력의 FP32 형식을 바꾸지 않는다. `optimizer_to_device()`는 AdamW의 noncapturable/nonfused scalar step을 CPU에 유지하고 모멘텀 텐서만 필요한 장치로 옮긴다.

## torch.compile 별도 실험

공식 PyTorch 문서는 Windows Intel XPU의 eager·AMP와 PyTorch 2.7 이후 `torch.compile`을 지원한다고 설명한다. Windows Inductor 문서는 C++ 도구체인 준비도 요구한다. 현재 환경에서 API와 Triton 모듈은 있지만 `cl`/`clang-cl`은 PATH에 없었다. 실제 CHROMATIC 모델의 15초 격리 실험에서는 Triton이 `driver.c`를 cp949로 읽다가 UnicodeDecodeError를 냈고 프로세스 종료도 상한 내 완료되지 않아 하위 프로세스까지 종료했다. 따라서 현재 환경에서 compile의 성공·성능은 확인되지 않았고 기본값에서 사용하지 않는다. Windows XPU 전체가 미지원이라는 주장도 하지 않는다.

- [공식 Intel GPU 시작 문서](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)
- [공식 Windows CPU/XPU compile 안내](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
- [원시 JSON](../reports/benchmarks/microbenchmark.json) / [CSV](../reports/benchmarks/microbenchmark.csv) / [JSONL](../reports/benchmarks/microbenchmark.jsonl)
- [추천](../reports/benchmarks/recommendation.json), [compile 환경](../reports/benchmarks/compile-environment.json), [격리 compile 실험](../reports/benchmarks/compile-probe.json)
