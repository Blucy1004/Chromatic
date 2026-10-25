# 모델 출처와 56/100 성과 감사

## 확인된 결론

기존 **22/100·45/100·56/100은 모두 동일한 ZERO 초기 챔피언 가중치**를 사용했다. Lichess 10,000개 후보를 사용한 성과가 아니다. 모델을 교체하지 않고 탐색 옵션을 바꾼 결과다. 기존 기록에는 파일 SHA-256이 있지만 프로세스 메모리에서 계산한 가중치 해시는 없으므로, 과거 프로세스의 실제 메모리 해시는 **미확인**으로 남긴다. 이번 재실행은 `zero model`로 실제 로딩된 텐서 해시를 별도로 확인한다.

감사 원시 자료: [models.json](reports/provenance/models.json). 새 모델을 학습하거나 기존 모델·배포본을 수정하지 않았다.

## 1. 정확한 모델 파일과 계보

### ZERO 초기 챔피언

- 기존 22/45/56 실행 경로: `/workspace/Chromatic/project-chromatic-memory/dist/default.zero`. 과거 드라이버의 `MODEL` 상수와 같은 실행 파일 해시를 사용하는 동시기 메타데이터로 경로를 연결했다. 과거 puzzles JSON 자체에는 경로가 기록되지 않았다는 한계는 남는다.
- 이번 교차 검증 경로: `/workspace/Chromatic/chromatic-shard-trainer/champion/default.zero`. 위 파일 및 `project-chromatic/dist/default.zero`, `project-chromatic/baseline/default.zero`, `project-chromatic/runs/bootstrap/model.zero`와 바이트가 동일하다.
- 모델 SHA-256: `f6a00902c23e7f32f96b028ac36464cc9012cd8774d5e6b0dee5e1c94e1c03c9`.
- 체크포인트: `/workspace/Chromatic/chromatic-shard-trainer/champion/model.pt`; `project-chromatic/runs/bootstrap/model.pt`와 동일하다.
- PT SHA-256: `99151f0c57212edd6786e2bbbafc1717bb15bee9a24341e69c5606c478bb4b84`.
- 아키텍처: `ZERO0001` little-endian FP32, 입력 1,280, 폭 64, residual Linear 블록 2개, Policy 출력 4,672, WDL 출력 3, 파라미터 394,179개. PT 체크포인트 포맷은 1이다. 엔진 버전과 가중치 포맷 버전은 별개다.
- 학습 규모: 보존된 Stockfish 교사 표본 **128개**, 학습 **200스텝**, 배치 32, seed 2026. `teacher.jsonl`의 실제 행 수와 SHA가 PT의 dataset 기록 및 `teacher.meta.json`에 일치한다.
- 교사 데이터 SHA-256: `d3dd6ce286c7c2421ba35dd3163b879e23328b486b251a28a170cfb210f50e77`.
- 교사 기록: Stockfish 19, 위치당 2,000 nodes, MultiPV 4; 기록된 교사 바이너리 SHA는 `45bc8e4969147db9c2eb533810637994619bff0eacc81ccfd9854394901bcbd0`. 해당 교사 바이너리로 독립 재학습한 결과를 주장하지 않는다.
- PT 텐서를 ZERO0001 순서로 다시 직렬화해 계산한 해시가 배포 모델 SHA와 정확히 일치한다. 학습 실행 자체를 다시 수행한 것은 아니다.

근거: [교사 메타데이터](../project-chromatic/runs/bootstrap/teacher.meta.json), [학습 기록](../project-chromatic/runs/bootstrap/model.train.json), [초기 릴리스 manifest](../project-chromatic/release-manifest.json).

### Lichess 10,000개 후보

- 모델 경로: `/workspace/Chromatic/chromatic-shard-trainer/candidates/positions-10000.zero`.
- 모델 SHA-256: `44957f9a35d534883ec92cf015a380dcb20b2fe6096150f687c3a0aa2f75e16a`.
- PT 경로: `/workspace/Chromatic/chromatic-shard-trainer/candidates/positions-10000.pt`.
- PT SHA-256: `427da59067d8acf1a799ad57ff8dbbceb327d070aeed4fb9fc12bb3911453143`.
- 아키텍처는 초기 챔피언과 동일한 ZERO0001 / 1,280–64 / residual 2 / 4,672+3 / 394,179개 파라미터다. PT 포맷은 2, trainer_schema는 1이다.
- 시작 체크포인트 해시가 위 초기 챔피언 PT의 `99151f…`와 정확히 일치한다. 가중치만 새로 무작위 초기화한 모델이 아니다.
- 새 학습 규모: **10,000개 처리 완료**, 샤드 1,000개 × 10, 샤드당 3 epochs, 배치 32, **1,149회 추가 업데이트**, seed 2026. 고정 validation/test 각각 1,000개는 학습 규모에 합산하지 않는다. Replay 재사용과 교사 표본 중복 가능성이 있으므로 총 고유 위치 수를 단순히 10,128로 단정하지 않는다.
- 원본 입력은 공식 Lichess eval의 로컬 60,000행 fixture이며, 현재 파일 SHA는 `feb08110ffcef588f05c30bb7ca93f5e19897e85c3d84678aff076f51dd84e2b`다. PT에 기록된 로컬 source hash와 일치한다. 실제 처리된 원본 행은 보고서상 25,728행이다. 60,000행 전체를 모두 학습한 모델이라고 표현하지 않는다.
- 보존된 `bundle-a181d36c81b94b22aa786094e626b6b8.manifest.json`의 commit_seq 56, step 1,149, positions_processed 10,000을 확인했다. 본체 PT·미러 PT·ZERO와 편의 사본 후보의 해시가 일치한다. PT 텐서의 재직렬화 해시도 ZERO와 일치한다.
- ARC의 cpu4-selected/XPU 후보와 혼동하지 않도록, `candidates/index.json`이 명시하는 원래 CPU 학습기의 `positions-10000` 후보를 선택했다. 기존 후보의 승격 여부는 바꾸지 않았다.

근거: [후보 index](../chromatic-shard-trainer/candidates/index.json), [학습 보고서](../chromatic-shard-validation/positions-10000/candidate/training-report.json), [검증된 상세 계보](reports/provenance/models.json).

## 2. 기존 22/45/56 실험의 실행 조건

세 실험의 공통 모델 경로·해시·아키텍처·학습 이력은 위 ZERO 초기 챔피언과 같다. 원본 실행 파일은 `/tmp/chromatic-overlord-release/chromatic_overlord`, SHA-256은 `b2bff565c06df56f9496ceb2186d557f282c66e1c6fb84ecbcbb9c690b810053`다.

| 기존 성과 | CounterexampleVerify | AdaptiveBudget | ForcingSearch | TacticalVerify | 제한 | 스레드 / seed |
|---|---|---|---|---|---|---|
| 22/100 | false | false | false | false | 100ms/문제 | 1 / 1 |
| 45/100 | false | false | false | true | 100ms/문제 | 1 / 1 |
| 56/100 | true | true | true | true | 100ms/문제 | 1 / 1 |

공통: CPuct 1.5, FPU 0, SIMD, sparse policy true, MateProofNodes 0, MateProofPlies 1, SharedTree true(스레드 1이라 직렬), VerificationNodes 2,048 / Depth 4 / Candidates 4, TreeMemoryMB 64, MaxNodes 100,000, noise/temperature 0. 과거 JSON에서 직접 보존된 해시/정답 수와 드라이버·엔진 소스로 복원한 옵션은 구분한다. 과거 실행 시점의 로딩 텐서 해시는 기록되지 않았다.

문제 파일: `chromatic-arc-trainer/fixtures/official-puzzles.csv`, SHA-256 `c46a5c5d83a27e4418a4a13aa196b4097c9aa863d76ce18386d61c8e628cecfb`. 게시된 첫 상대 수를 적용한 뒤 기대 응수와 일치하는지 평가했다. 다른 응수도 승리 수일 수 있어 정확 응수율을 완전한 전술 정답률이나 Elo로 해석하지 않는다.

과거 원시 기록: [puzzles-final.json](reports/puzzles-final.json). 새 원본 바이너리 재실행: [CSV](reports/provenance/linux-original-binary-replay/summary.csv). 같은 바이너리로 3회 실행해 기본 `[22,22,22]`, 전술 `[45,45,45]`, 조합 `[56,56,55]`를 관측했다. **56은 재현됐지만 고정 벽시계 예산의 모든 실행이 항상 56이라는 뜻은 아니다.**

## 3. 실제 로딩된 모델 해시를 확인하는 방법

새 읽기 전용 명령 `zero model`은 검증된 메모리의 header·Linear weights·bias를 ZERO0001 순서로 재구성해 SHA-256을 보고한다. 나중에 같은 경로의 파일을 다시 읽어 얻은 해시가 아니다. 각 셀의 시작과 끝에서 이 값이 실제 입력 파일의 SHA와 같고, 파일/바이너리 자체도 변경되지 않았는지 확인한다. 메타데이터에는 경로, 아키텍처, 바이너리 해시, 옵션, OS, CPU affinity, 스레드, seed, 제한 시간이 함께 저장된다.

SHA 구현은 표준 empty / abc / million-a 벡터와 알려진 챔피언 파일 해시로 검사한다. 해시 명령 추가 전후의 3개 탐색 설정 × 2개 위치에서 고정 노드 결과가 완전히 같음을 확인했다. 탐색 알고리즘이나 가중치를 변경하는 계측은 아니다.

새 Linux 감사 바이너리: `/tmp/chromatic-overlord-audit/chromatic_overlord`, SHA-256 `8d61aaacf530f2940360bb84683e584e493de31b0c4410170dba708f04541414`. 실제 로딩 digest는 초기 챔피언 `f6a009…`, 후보 `44957f…`로 각각 파일/체크포인트 재직렬화 digest와 일치한다. Windows 바이너리·실제 절대 경로·digest는 Windows 전용 원시 결과에 별도로 보관한다.

## 4. Linux 교차 검증

동일 하드웨어/CPU 집합, 100개 문제, 100ms, seed 1, 스레드 1, 같은 바이너리로 각 셀 3회 실행했다. 반복은 실행 시간 변동을 보기 위한 것이며 300개의 독립 문제로 세지 않는다.

| 모델 | 기본 MCTS | 전술 검증 | OVERLORD 조합 |
|---|---|---|---|
| ZERO 초기 챔피언 | 22 / 22 / 22 | 45 / 45 / 45 | 56 / 56 / 56 |
| Lichess 10,000개 후보 | 34 / 35 / 35 | 50 / 50 / 50 | 50 / 51 / 52 |

- 탐색 개선 효과(같은 모델, 중앙값): 초기 챔피언에서 전술 +23문제, 조합 +34문제; 10,000개 후보에서 전술 +15문제, 조합 +16문제.
- 모델 개선 효과(같은 탐색, 중앙값): 기본 MCTS +13문제, 전술 +5문제, 조합 **−5문제**. 더 학습한 모델이 모든 탐색 설정에서 단조롭게 좋아진다고 결론 내릴 수 없다.
- 시간제어에 따른 탐색량과 후보 검증 완료 정도가 달라져 상호작용이 존재한다. 고정 문제 표본의 지표이며 일반 기력/Elo 또는 후보 승격 근거로 확장하지 않는다.

원시 [metadata](reports/provenance/linux-matrix/metadata.json) / [CSV](reports/provenance/linux-matrix/summary.csv)와 셀별 `.json.gz`에 실제 로딩 digest와 모든 응수를 보존했다.

## 5. Windows 교차 검증

Windows x64/MSVC Release에서 실제 실행했다. 동일 100문제·100ms·seed 1·스레드 1, 셀마다 3회다.

| 모델 | 기본 MCTS | 전술 검증 | OVERLORD 조합 |
|---|---|---|---|
| ZERO 초기 챔피언 | 22 / 22 / 22 | 45 / 45 / 45 | 50 / 49 / 50 |
| Lichess 10,000개 후보 | 34 / 35 / 34 | 48 / 48 / 48 | 47 / 47 / 47 |

Windows 실행 파일 SHA는 `de9786280fab55b47a2eccb51f030a2b85b49d097a6b8111fff0e9e2d94df451`, MSVC 버전은 19.44.35229.0이다. 두 모델의 실제 로딩 digest는 각각 위 `f6a009…`, `44957f…`와 일치했다. 원래 파일 경로는 Windows [metadata.json](reports/windows-native/40146054219-1/matrix/metadata.json)의 `model_paths`와 각 셀의 `loaded_model.path`에 있다.

Windows에서 모델만 바꾼 중앙값 차이는 기본 +12 / 전술 +3 / 조합 −3문제, 탐색만 바꾼 차이는 ZERO 전술 +23 / 조합 +28, 후보 전술 +14 / 조합 +13이다. Windows에서는 56이 재현되지 않았으며 Linux/Windows 결과를 합치지 않는다. 서로 다른 머신이므로 OS 간 차이를 동일 하드웨어에서의 OS 효과로 해석하지 않는다. 빌드·프로토콜·시간·규칙 검사와 원시 Windows 로그는 [WINDOWS_VALIDATION.md](WINDOWS_VALIDATION.md)에 있다.

## 보존과 한계

초기 챔피언, 후보, 원래 배포 바이너리 및 다른 기존 가중치는 변경하지 않았다. 신규 기능은 기본 비활성, Threads 기본값은 1이며 어떤 후보도 승격하지 않았다. 과거 프로세스 메모리 해시, 실제 교사 바이너리의 재실행, 전체 원격 Lichess DB에 대한 독립 검증은 확인하지 않은 사실로 남긴다. 모델 파일·현재 PT 텐서·보존된 학습/원본 표본의 결합은 파일 해시로 확인했다.
