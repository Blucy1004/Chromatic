# PROJECT ZERO 0.1 검증 결과

실행일: 2026-10-24, Asia/Seoul. C++20/MSVC 19.51, CMake 4.4/Ninja, Python 3.12.14, PyTorch 2.14.1+cpu, python-chess 1.999(chess 1.11.2), NumPy 2.5.4 환경에서 실행했습니다. Windows x64 CPU 실행이며 CUDA 경로는 검증하지 않았습니다.

## 요청된 네 가지 검증

| 항목 | 실행 결과 |
|---|---|
| perft | 아래 5개 기준 위치 모두 일치, CTest 통과 |
| UCI | 초기화, position, nodes, movetime, stop, infinite 도중 ready, 0초 시계, 종료 위치, 불법 수 오류 처리 통과 |
| 모델 추론 일치 | 무작위 초기 모델 52개 위치 및 실제 학습 후보 100개 위치에서 모든 정책 logits/WDL 비교 통과 |
| 체크포인트 복구 | 연속 학습과 4스텝 저장 후 12스텝까지 재개 결과 동일. 최신 파일 손상 시 이전 8스텝 파일에서 복구해 16스텝까지 재학습한 결과도 연속 학습과 동일 |

복구 동등성은 가중치만 비교하지 않고 optimizer, Torch 샘플러 generator, Python/NumPy/Torch RNG도 비교했습니다. 데이터가 달라진 상태의 resume 거부도 확인했습니다. 모델 파일을 잘라서 잘못 로드했을 때 C++이 기존 모델을 보존하는지, 실행파일 옆의 `default.zero`를 자동으로 읽는지도 확인했습니다.

## Perft 수치

| 위치 | 깊이 | leaf nodes |
|---|---:|---:|
| 시작 위치 | 5 | 4,865,609 |
| Kiwipete | 4 | 4,085,603 |
| CPW position 3, 룩/폰 엔드게임 | 5 | 674,624 |
| CPW position 4, 승격/캐슬링 | 4 | 422,333 |
| CPW position 5, 전술/승격 | 4 | 2,103,487 |

정확한 FEN은 `tests/core.cpp`에 포함되어 있습니다. 반복, 스테일메이트, 메이트의 50수 판정보다 우선 처리, 앙파상 발견 체크, 승격 action 충돌, PUCT 메이트의 역전파 부호, 트리 노드 상한에서 방문 수 보존도 검사했습니다.

## 독립 대조와 추론 오차

Python chess 규칙 구현과 C++ 구현을 1,008개 위치에서 대조했습니다. 합법 수 집합, 각 수의 정책 인덱스, FEN, 1,280개 입력값과 종료 여부가 일치했습니다. 흑/백 승격과 반복 이력도 포함했습니다.

초기 모델 52개 위치에서 최대 절대오차는 정책 `3.58e-7`, WDL `5.94e-8`였습니다. 실제 학습한 후보는 100개 위치에서 정책 약 `5.72e-6`, WDL 약 `4.77e-7`였으며 허용치 `2e-5`를 통과했습니다. 정확한 최종 값과 바이너리 해시는 `validation/trained-parity.json`을 확인하세요.

## 실제 부트스트랩과 자가학습 실행

1. 공식 Stockfish 19를 외부 교사로 사용해 128개 위치를 수집했습니다. 위치당 2,000노드, 최대 MultiPV=4, seed=2026으로 실행했습니다. 교사 ID/SHA와 데이터 SHA는 `runs/bootstrap/teacher.meta.json`에 있습니다.
2. 폭 64, residual 층 2 모델을 200스텝 학습했습니다. 마지막 minibatch의 정책 loss 약 0.719, WDL loss 약 0.103이었습니다. 이는 학습 자료에서의 손실이며 별도 홀드아웃 성능은 아닙니다.
3. 시작 위치에서 교사 없이 4판 자가대국했습니다. 64 시뮬레이션/수, 최대 512플라이로 실행하여 3판 완료, 1판 중단, 190개 학습 위치를 얻었습니다. 완료 3판은 모두 3회 반복 무승부였으며, 잘린 1판은 학습에서 제외했습니다.
4. 이 **자가대국 190개 위치만** 사용해 후보를 100스텝 학습했습니다.
5. 두 오프닝 쌍에서 색을 바꾸어 4판 대결했습니다. 후보 점수 0.5, 신뢰하한 0.0으로 승격 기준 0.55에 미달하여 초기 챔피언을 유지했습니다.

최종 엔진으로 수행한 결과는 `runs/release-validation/state.json`, `generation_0000/selfplay/summary.json`, `candidate.train.json`, `arena.json`에 보존했습니다. 이전 개발 실행은 배포 ZIP에 포함하지 않습니다. 반복 실행이 완료 상태를 유지하는지도 확인했습니다.

이 실행은 학습 파이프라인과 승격 절차가 작동한다는 근거입니다. 작은 네트워크, 128개 교사 위치, 4판 자가대국과 4판 대결로는 경기력 향상이나 강한 체스 실력을 입증할 수 없습니다. Elo 수치는 측정하지 않았습니다. 실제로 반복 무승부 편향이 관찰되어 후속 개발에서 결과 다양성과 전술 능력 개선이 필요합니다.

## 재실행

```powershell
.\validate.ps1
.\run.ps1 verify --executable build\project_zero.exe --checkpoint runs\release-validation\generation_0000\candidate.pt --output validation\trained-parity.json --positions 100
```

일반 Windows 환경에서 실행하세요. 이번 도구 환경에서는 Windows UCI 클라이언트의 통신과 기본 임시 폴더 접근이 제한되어 초기 검증 실행이 막혔습니다. 테스트의 임시 파일은 프로젝트 내부 폴더로 옮겼고, UCI 외부 클라이언트 검증은 정상적인 Windows 접근 권한으로 실행했습니다. 최종 `validate.ps1`에서 CTest와 통합 테스트 5개가 모두 통과했습니다.
