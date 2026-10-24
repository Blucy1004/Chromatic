# ARC 학습 정확성·복구 검증 계약

기존 엔진·챔피언·이전 CPU 실험을 입력으로만 읽는다. 새 테스트 실행과 보고서는 별도 ARC 폴더 및 명시한 새 테스트 디렉터리에 저장한다. `tools/validate_arc.py`는 네트워크를 사용하지 않는다. 실제 GPU를 쓰는 검사는 명시적으로 `--device xpu`를 지정한다.

## 수치 비교와 장치 이동

`--phase numeric`은 고정 normalized JSONL에서 최대 100개 레코드만 읽는다. 기본 32개 동일 입력·동일 가중치·동일 confidence 가중 loss에 대해 CPU FP32와 지정 장치의 Policy logits, WDL 확률, backward gradient, gradient clip 1 이후 AdamW 한 단계 가중치를 비교한다. XPU 구간의 시작과 끝에는 `torch.xpu.synchronize()`를 적용한다.

최종 수치 입력은 `reports/numeric-inputs.jsonl`의 32개다. 고정 test 1,000개에서 원문을 바꾸지 않고 weak 오프닝 앵커 16개와 positive Value 가중치 16개(초반6·중반5·종반5)를 결정적으로 골랐다. `numeric-inputs.meta.json`에 원본 SHA·행 번호·ID·phase·가중치·선택 파일 SHA를 기록했다. `numeric-native-inputs.jsonl`은 이 배치의 고른 10개 인덱스로 5weak·5positive 및 세 단계를 포함한다. 입력 준비 도구는 JSON 파일만 처리하며 학습·GPU·다운로드를 실행하지 않는다.

Policy와 WDL head 각각의 raw gradient norm 및 업데이트 norm을 따로 보고한다. 두 head에서 유한한 nonzero raw gradient를 실제로 관측해야 `both_heads_gradients_exercised`와 최종 `passed`가 참이 된다. 전체 Value 가중치가 0인 입력처럼 backward 검증 범위가 부족하면 `validation_coverage_weak=true`, `passed=false`를 기록하고 중단한다. 초기 고정 test의 앞 32개가 모두 Value 가중치 0 앵커였던 기존 자료는 `reports/pre-mixed/`에 한계와 함께 보존했다. 그 자료의 WDL forward 비교와 clipping 결함 증거는 유효하지만 WDL backward·업데이트를 직접 검증한 자료로 해석하지 않는다.

미리 정한 FP32 한계는 Policy logits 절대 오차 1e-4, WDL 확률 1e-5, gradient 1e-4, 업데이트된 가중치 1e-4, 상대 loss 1e-5다. BF16은 별도 실험이며 해당 한계는 각각 0.25, 0.02, 0.02, 0.0021, 0.05다. BF16의 느슨한 한 단계 한계만 통과했다고 기본값으로 채택하지 않으며, 부모 벤치마크의 여러 학습 단계와 고정 검증 손실도 함께 판단한다. 장치 간 비트 일치를 요구하지 않는다.

추가로 각 장치에서 복사한 실제 gradient의 CPU float64 제곱합 norm을 독립 참조로 계산한다. clip 반환 norm의 오차는 `max(1e-5, reference × 1e-5)` 이내, clip 후 실제 norm과 예상 norm 차이는 1e-5 이내여야 한다. Arc 140T / PyTorch 2.14.1+xpu에서 `clip_grad_norm_`의 foreach 기본/True 경로가 1.901946이어야 할 norm을 2.163705로 계산해 clip 후 norm이 0.879022가 되는 오류를 실제로 발견했다. `foreach=False`와 독립 square-sum은 참조와 7.1e-8 이내로 일치했다. [진단 원시 보고서](../reports/clip-diagnostic.json)를 보존하고 이후 XPU 수치 검사는 non-foreach 경로를 사용한다. CPU의 기존 clipping 경로는 유지한다. 잘못된 clip 설정에서의 중단 재개 일치는 올바른 gradient 연산의 증거가 아니므로 수정한 실행의 복구 검사를 별도로 수행했다.

기존 CPU 체크포인트의 모델·Optimizer를 지정 장치에 로드해 한 단계를 수행하고, 시험한 장치의 모델·Optimizer를 모두 CPU 텐서로 복사한 뒤 CPU Optimizer가 다음 단계를 수행하는지도 검사한다. 새 실행의 `InitialCheckpoint`가 모델 가중치만 불러오는 동작과, 동일 실행의 전체 학습 상태를 복구하는 Resume는 구분한다. 현재 실행 설정의 장치를 바꾼 채 동일 ledger를 재개할 수 있다는 주장은 하지 않는다.

XPU 난수 상태는 CPU uint8 텐서로 저장하며, 저장→장치 난수 생성→복원→다시 난수 생성 결과가 정확히 같은지도 검사한다. 실제 학습 체크포인트는 `rng_xpu`를 포함하고 모델·Optimizer 텐서는 CPU로 저장해 XPU가 없는 프로세스에서도 읽을 수 있어야 한다.

```powershell
python .\tools\validate_arc.py --phase numeric --device xpu --precision fp32 `
  --records .\reports\numeric-inputs.jsonl --output .\reports\numeric-xpu-fp32.json
```

## 기존 C++ 추론·내보내기

`--phase native`는 지정한 신뢰할 수 있는 PT를 CPU에서 읽고 별도의 `.zero`로 내보낸다. 보유한 CHROMATIC 메모리 개선 엔진에 SIMD·sparse policy 옵션을 적용하고, 동일 입력의 features, 합법수/action ID, Policy logits, WDL 확률을 비교한다. 허용 절대 오차는 features 1e-7, Policy/WDL 2e-5다. 입력 PT와 엔진의 SHA를 전후 비교한다. 이 검사는 엔진 바이너리나 챔피언 파일을 쓰지 않는다.

```powershell
python .\tools\validate_arc.py --phase native --checkpoint <후보.pt> `
  --records .\reports\numeric-native-inputs.jsonl --maximum-records 10 `
  --engine ..\project-chromatic-memory\dist\chromatic-memory.exe `
  --output .\reports\native-candidate.json
```

## 동일 장치 강제 종료·Resume

`--phase recovery`는 공식 로컬 fixture로 100개 포지션·샤드 크기 25개(총 4샤드)·2에폭의 새 실행을 만든다. 중단 없는 기준 실행과 다음 실제 자식 프로세스 시험을 순차로 비교한다.

1. 완료 후 Resume: 추가 학습과 새 모델 커밋이 없어야 한다.
2. 체크포인트 파일이 쓰인 뒤 SQLite 확정 전 강제 종료.
3. SQLite 확정 직후 강제 종료.
4. 이전 성공 샤드 삭제 직전 강제 종료.
5. 새 테스트 실행 안의 최신 본체만 손상시켜 미러 복구, 이어 미러도 손상시켜 이전 정상 체크포인트 복구.

모델·Optimizer·Scheduler·CPU/local-generator/Python/NumPy 및 XPU RNG·Replay·배치 순열·에폭·offset·진행 상태·고정 세트 ID/SHA·소스 cursor를 **정확 비교**한다. 시간·출력 디렉터리·새 bundle UUID는 비교하지 않는다. 완료 샤드 커밋이 정확히 네 개인지도 검사한다. 서로 다른 장치 실행끼리 이 비트 일치 시험을 비교하지 않는다.

```powershell
python .\tools\validate_arc.py --phase recovery --device xpu --precision fp32 `
  --validation-mode selected --profile-resume `
  --directory <새로운-빈-시험폴더> --output .\reports\recovery-selected-xpu.json
```

시험 자식 프로세스의 메모리 기본 한도는 4GiB이며 `--max-memory-gb`로 명시적으로 조정할 수 있다. 공유 GPU 메모리와 프로세스 committed memory의 관계는 하드웨어·드라이버에 따라 달라지므로 실제 보고서의 한도와 메모리 표본을 함께 확인한다.

검증 도구의 `--validation-mode` 기본값은 기존 호출 호환성을 위해 `full`이다. `selected` 검사 결과는 기존 full 결과와 별도 파일로 저장한다. `--profile-resume`은 기준 학습을 계측하지 않고 재개한 자식에만 `--profile`을 전달한다. 완료 Resume·kill Resume·미러/이전 버전 Resume의 `checkpoint_recovery` exclusive 초와 calls를 추출하고, 매번 profile JSON을 새 파일로 복사해 다음 재개가 덮어쓰지 않도록 보존한다.

실제 XPU clipping 회귀 테스트는 `CHROMATIC_RUN_XPU_TESTS=1`을 설정한 뒤 `pytest tests/test_arc_recovery.py`로 명시적으로 켤 수 있다. 일반 CPU 테스트에서는 GPU를 사용하지 않도록 생략된다. 이 검사는 기존 모델과 같은 gradient 텐서 크기에서 XPU clipping 결과를 독립 CPU float64 제곱합 norm과 비교한다.

단위 테스트는 실제 GPU나 원격 데이터를 사용하지 않는 작은 범위다. 실제 XPU·수치·복구 명령의 실행 결과는 별도 원시 JSON 보고서로 확인한다. 체크포인트 본체·미러·필요한 이전 샤드가 모두 손실된 경우나 SQLite 자체의 모든 사본이 손실된 경우까지 복구할 수 있다고 주장하지 않는다.

## 이 환경에서 실제 완료한 결과

Python 3.11.9 / PyTorch 2.14.1+xpu 전용 환경에서 수정 후 [CPU FP32](../reports/numeric-cpu-fp32.json), [XPU FP32](../reports/numeric-xpu-fp32.json), [XPU BF16](../reports/numeric-xpu-bf16.json) 수치 검사를 실행했다. 세 조건 모두 원본 모델 보존, 기존 Optimizer 호환, CPU 텐서 PT의 실제 저장·재로드와 CPU 다음 단계까지 통과했다. XPU 두 조건은 장치 RNG 복원도 정확 일치했다.

최종 mixed 입력의 수정된 XPU FP32는 CPU 대비 Policy logit 최대 오차 4.77e-6, WDL 2.38e-7, gradient 1.04e-7, 기본 AdamW 한 단계 가중치 1.56e-7이었다. CPU의 raw Policy/WDL head norm은 각각 1.1479783 / 1.1270671이고 XPU도 각각 1.1479785 / 1.1270672로 양쪽 head를 실제로 검증했다. XPU clipping norm의 CPU float64 참조 오차는 5.90e-8 이내였다. BF16의 한 단계 가중치 오차는 0.00199823으로 선언한 0.0021 한계에 가깝다. 따라서 BF16 한 배치 통과만으로 장기간 학습 품질이나 기본값 채택을 보증하지 않는다.

CPU와 수정된 XPU FP32에서 각 5개 실제 복구 시나리오를 모두 통과했다. 설정은 Prefetch 2, CacheMode auto, BatchSize 32, CPU threads 1, 100포지션·2에폭·샤드 크기 25개(총 4샤드)다. 각각의 기준 실행은 업데이트 8단계다. XPU-only clipping 수정 전후로 CPU 연산 경로는 같으며, 최종 XPU suite는 수정 후 새 디렉터리에서 전부 다시 실행했다.

- [CPU 실제 복구 결과](../reports/recovery-cpu.json)
- [수정 후 XPU 실제 복구 결과](../reports/recovery-xpu.json)
- [장치별 요약 CSV](../reports/recovery-validation-summary.csv), [전체 원시 로그 SHA·위치 JSON](../reports/recovery-validation-summary.json)
- [실제 XPU clipping을 포함한 5개 회귀 검사 XML](../reports/arc-xpu-regression.xml)

원시 복구 콘솔 로그 22개를 `reports/recovery-logs/`에 별도 보존했다. 수정 전 XPU 수치·복구 보고서는 `reports/pre-clip-fix/`에 진단 자료로만 남기며 최종 정확성 근거로 사용하지 않는다.

이후 최종 selected 전처리 경로에서도 CPU·XPU 각각 5개 복구 시나리오를 새 디렉터리에서 모두 다시 통과했다. 두 장치 각각의 기준 `.zero` SHA는 기존 full 경로와도 동일했다. selected 원시 로그 22개와 재개별 profile snapshot 12개는 `reports/recovery-selected-logs/`, `reports/recovery-selected-profiles/`에 보존했다. checkpoint bundle 로드·검증의 exclusive span 중앙값은 CPU 0.02537초, XPU 0.02765초다. 전체 Resume 프로세스의 Python/장치 초기화·모델 배치·고정 데이터 준비·검증 시간은 이 span과 구분하여 별도 wall time으로 기록했다.

- [selected CPU 복구](../reports/recovery-selected-cpu.json), [selected XPU 복구](../reports/recovery-selected-xpu.json)
- [selected 복구 요약과 SHA 색인](../reports/recovery-selected-summary.json), [시나리오 CSV](../reports/recovery-selected-summary.csv)
- [복구 exclusive span과 프로세스 wall time CSV](../reports/checkpoint-recovery-timings.csv)

CPU4 selected, XPU FP32 selected, XPU BF16 full에서 각각 10,000개를 학습한 실제 후보 3개를 C++ 엔진과 비교했다. mixed 배치에서 고른 10개씩 총 30개 조합 모두 통과했다. 최대 오차는 features 2.68e-10, legal Policy logits 9.57e-7, WDL 확률 1.19e-7이었다. 엔진과 원본 PT의 SHA도 보존됐다. [네이티브 검증 요약](../reports/native-summary.json)과 개별 `native-*.json/.log`에 입력 ID·각 오차·PT/엔진/내보내기 SHA가 남아 있다.
