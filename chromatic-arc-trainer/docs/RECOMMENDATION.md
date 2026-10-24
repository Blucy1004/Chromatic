# 전체 파이프라인 추천과 적용 범위

`tools/recommend_training.py`는 완료된 전체 실행의 벽시계 시간과 고정 검증 손실을 사용한다. 이미 인코딩한 한 배치의 step 속도만으로 장치나 배치 기본값을 바꾸지 않는다. 프로그램은 학습·다운로드·GPU 연산을 시작하지 않으며 원본 결과를 수정하지 않는다.

입력은 `metadata`와 `results`가 있는 JSON, JSON 배열 또는 한 행에 한 완료 결과를 담은 JSONL이다. JSONL에 공통 설정이 없으면 `--metadata`로 별도 JSON을 지정한다. 명령 기록과 후보 manifest·학습 보고서·완료 샤드의 SHA는 읽기 전용으로 확인할 수 있다. 메모리 표본은 추천 점수에 사용하지 않는다. 이전 실행기의 부모 프로세스 RSS 표본을 실제 학습 프로세스 메모리라고 취급하지 않기 위해서다.

필수 증거는 실제 하드웨어/모델 signature, 원본/초기 PT SHA, 모델 구조, 포지션 수, 에폭·시드·Replay·샤드 크기·학습률·체크포인트 주기, 고정 holdout, 처리한 소스 위치·품질 결정, 완료 샤드 SHA, 실제 업데이트 및 학습 샘플 수다. 행의 `signature` 또는 `signature_key`, 혹은 metadata의 `signature`가 `microbenchmark.json`의 실제 측정 signature와 일치해야 한다. 과거 다른 장비의 행에 현재 signature를 자동으로 붙이지 않는다. 측정한 동일 세션이라는 사실을 확인한 운영자가 metadata에 기록한다.

기본 비교 조건은 BatchSize 32, 10,000포지션, 성공한 독립 반복 3회다. 동일 자료의 CPU 1스레드 FP32를 기준으로 Policy·WDL 검증 loss가 **각각 2% 이내**여야 한다. 하나의 실패나 NaN을 좋은 반복 평균으로 숨기지 않는다. 실패·timeout·미완료·프로파일링 실행·장치/precision fallback은 성능 선택에서 제외한다. 업데이트 수나 실제 처리 샘플 수가 다르면 같은 Epochs라는 이유만으로 공정한 가속 비교에 넣지 않는다.

전체 CPU 스레드 추천은 품질·비교 조건을 통과한 CPU 후보의 전체 시간 중앙값으로 정한다. `auto_device_profiles`는 **batch/thread/precision/cache/prefetch/optimizer/accumulation/transfer/ValidationMode가 모두 같은 CPU와 XPU 증거가 있을 때만** 생성한다. 다른 조건을 섞어 CPU→XPU 전환을 추천하지 않는다. 반복 수, 최소/최대 시간과 변동 계수도 함께 기록한다.

이번 1,000개 실험에서 batch 256·512의 WDL loss 약 0.869·0.975는 BatchSize 32 기준 약 0.788보다 나빴다. 이 후보들은 loss gate와 업데이트 수 조건을 통과하지 못하므로 microbenchmark가 빨라도 기본 배치로 추천하지 않는다. 1,000개 결과를 10,000개 또는 다른 모델 크기의 추천으로 확대하지 않는다.

```powershell
python .\tools\recommend_training.py `
  --results <완료행.jsonl> --metadata <측정-signature와-공통설정.json> `
  --signature-file .\reports\benchmarks\microbenchmark.json `
  --positions 10000 --batch-size 32 --min-repeats 3 `
  --output .\reports\benchmarks\pipeline-recommendation.json
```

기본은 검토 보고서일 뿐 자동 적용할 수 없다. 완료 결과를 확인한 뒤 같은 명령에 `--register`를 명시하면 통과한 증거를 등록한다. `--min-repeats`는 운영자가 명시적으로 바꿀 수 있으며 실제 사용한 횟수를 보고서에 남긴다. loss 허용치는 2%를 넘길 수 없다.

`shardtrain.recommendation.read_validated(path, signature, constraints)`는 등록·자체 해시·원본 증거 파일 SHA·현재 signature·정확한 조건을 검사하고 맞는 장치 profile 또는 `None`을 반환한다. 설정을 직접 변경하지 않는다. 최소 runtime 조건은 `positions`, `batch_size`, `cpu_threads`, `precision`, `cache_mode`, `prefetch`다. 원본/초기 SHA 및 에폭·시드·Replay·샤드 크기·학습률·체크포인트 주기·optimizer·accumulation·transfer·ValidationMode도 호출자가 전달한 값은 모두 일치해야 한다. 데이터 범위를 넘어 추천을 적용하지 않으려면 이 값들을 함께 전달하고, 원본 SHA를 모르는 원격 입력에는 등록 추천을 적용하지 않는다.

명시적 `device=cpu` 또는 `device=xpu` 요청은 이 함수가 바꾸지 않는다. 다른 GPU·드라이버·PyTorch·모델 구조에서는 signature가 달라져 기존 추천을 거부한다. 자료가 바뀌거나 증거 파일이 손상돼도 추천을 거부한다. 추천 통과는 성능 설정에 대한 제한된 근거이며 모델 승격이나 체스 기력 향상을 뜻하지 않는다.
