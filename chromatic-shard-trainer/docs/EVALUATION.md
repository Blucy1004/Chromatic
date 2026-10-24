# 후보 모델 평가

평가 도구는 학습 ledger를 읽기 전용으로 열고, 최신 committed manifest의 체크포인트·복제본·네이티브 모델을 검증한다. 엔진 코드와 챔피언 모델을 수정하지 않는다. 같은 체크포인트를 다시 평가하면 동일 설정의 대국 진행 파일을 재사용한다.

```powershell
.\work\.venv\Scripts\python.exe outputs/chromatic-shard-trainer/tools/evaluate_candidate.py `
  --data-directory outputs/chromatic-shard-validation/positions-10000/data `
  --checkpoint-directory outputs/chromatic-shard-validation/positions-10000/candidate `
  --engine outputs/project-chromatic-memory/dist/chromatic-memory.exe `
  --champion-pt outputs/chromatic-shard-trainer/champion/model.pt `
  --champion-zero outputs/chromatic-shard-trainer/champion/default.zero `
  --fixtures-directory outputs/chromatic-shard-trainer/fixtures `
  --pairs 2 --milliseconds 10 --max-plies 128
```

세 가지 소규모 학습이 완료된 뒤 공통 holdout까지 비교하려면 다음 명령을 사용한다.

```powershell
.\work\.venv\Scripts\python.exe outputs/chromatic-shard-trainer/tools/evaluate_stages.py
```

`candidate/evaluation-report.json`과 버전별 `candidate/evaluations/<id>/`에 결과를 저장한다. 대국의 모든 수, 실제 수당 시간, 방문 수, PGN, 미완료 사유를 남긴다. `common-holdout.json` 및 CSV는 10,000개 학습 실행의 고정 validation/test를 모든 후보와 챔피언에 적용하고, 각 실행의 영속적인 학습 가족 목록과 교집합이 없는지 검사한다. 각 단계 자체 validation의 크기가 다르므로 단계별 loss를 직접 연결해 개선 그래프로 사용하지 않는다.

평가 항목은 confidence 가중 Policy/WDL 교사 라벨 loss, 시작 위치와 초기 20개 자식의 WDL·정책·부호 반전 잔차, 별도 실제 대국 결과의 Brier/NLL/ECE, 공식 CC0 퍼즐 20개의 공개 첫 응수 일치율, 같은 엔진과 검색 설정의 제한 시간 교환 대국이다. 실제 대국 결과의 보정 평가는 게임별 총 가중치를 1로 하며, 인간 대국의 결과를 완벽한 플레이의 이론적 WDL이라고 해석하지 않는다. 퍼즐의 다른 수가 이기는 경우도 있어 첫 응수 일치율만으로 전술 능력을 단정할 수 없다.

엔진은 `CPuct=1.5`, `FPUReduction=0`, `TreeMemoryMB=64`, `MaxNodes=1000000`, `MateProofNodes=0`, SIMD 추론, sparse policy를 사용한다. 검색 스레드는 하나이며 Windows affinity mask 2를 사용한다. 전술은 위치당 25ms, 기본 대국은 수당 10ms·2쌍·최대 128플라이이다. 모델과 동일 설정만 바꾼다. 불법 수와 엔진 오류를 따로 기록하고, 플라이 상한의 미완료 대국은 `*`이며 무승부 점수를 부여하지 않는다. 소규모 실행으로 Elo를 입증하지 않는다.

승격은 자동으로 수행하지 않는다. 충분한 고정 holdout, 원본 프레임 검증, 독립 보정 및 누수 검사, 비악화 지표, 엔진 오류 없음, 완료 대국 64개 이상과 95% 하한 등 엄격한 기준을 모두 충족해도 수동 검토 가능 여부만 표시한다. 이 도구에는 챔피언 덮어쓰기 경로가 없다. 새 데이터의 가족 누수만 검사할 수 있으며, 보존된 챔피언의 과거 학습 이력을 역으로 복원했다고 주장하지 않는다.
