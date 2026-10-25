# Windows 네이티브 검증 — 2026-10-25

**Windows x64/MSVC Release 빌드와 실제 엔진 실행을 확인했다.** Linux 실행 결과로 대체한 판정이 아니다. 기능 기본값과 챔피언은 변경하지 않았다.

## 실제 실행 환경과 바이너리

- [Windows 원본 실행](https://github.com/Blucy1004/Chromatic/actions/runs/40146054219): GitHub Actions `windows-2022`, Windows Server 2022 `10.0.20348`, AMD64.
- Visual Studio 2022 Enterprise / MSVC **19.44.35229.0**, CMake `-G "Visual Studio 17 2022" -A x64`, `--config Release`. 상세 [compiler record](reports/windows-native/40146054219-1/build.json)와 [빌드 로그](reports/windows-native/40146054219-1/build.log)를 보존했다.
- 검증한 엔진 소스 커밋: `4ae59930d84de76d5d31333e2f0e363bec9795bf`. 이후 변경은 출처/검증 문서, 결과 전달 및 문서 생성 오류 수정이며 탐색/추론 C++는 같다.
- 원래 Windows 실행 경로: `D:\a\Chromatic\Chromatic\build-windows-msvc\Release\chromatic_overlord.exe`.
- 실행 파일 SHA-256: `de9786280fab55b47a2eccb51f030a2b85b49d097a6b8111fff0e9e2d94df451`.
- [원본 Windows 산출물](https://github.com/Blucy1004/Chromatic/actions/runs/40146054219/artifacts/12580951939)에서 받은 바이너리도 위 SHA와 일치하고 PE signature 및 Machine `0x8664`(AMD64)를 확인했다. 후보 사본은 `/workspace/scratch/overlord-windows-release-40146054219/chromatic_overlord.exe`에 있다. 기존 dist 파일은 교체하지 않았다.
- CPU 식별: `AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD`, 모든 교차 검증 셀의 CPU affinity는 `[0,1]`, 검색 스레드는 1이다. Windows/Linux는 서로 다른 머신이므로 OS 간 차이의 원인을 OS만으로 단정하지 않는다.

## 네이티브 검사 결과

- [CTest 원시 로그](reports/windows-native/40146054219-1/ctest.log): **9/9 통과**. Perft/보드 규칙, 가치 부호, 메모리 통계 보존, 병렬 탐색, 중단, 메이트/스테일메이트/반복/50수, 실제 모델 및 로딩 identity 회귀 검사를 포함한다.
- [실제 UCI 프로토콜 기록](reports/windows-native/40146054219-1/platform/protocol.json): **41개 검사 통과**. 두 모델의 `uci`/`isready`/`go`/`stop`/`quit`를 실행했다.
- SANABI에 해당하는 CounterexampleVerify, Laugh Tale에 해당하는 AdaptiveBudget, ForcingSearch, TacticalVerify를 각각 true/false로 전환해 합법적인 응수를 확인했다. Threads 1↔2, 전체 기능+병렬 실행 및 탐색 중 quit 종료 코드 0도 확인했다.
- 실제 기존 CHROMATIC 가중치는 ZERO 초기 챔피언과 바이트 동일한 `chromatic-shard-trainer/champion/default.zero`로 로딩했다. 10,000개 후보도 별도로 로딩했고 각 경우 `zero model`의 실제 텐서 digest가 입력 파일 SHA와 일치했다.

### 시간제어 실측

아래는 모델 로딩을 마친 뒤 bestmove 수신까지의 지연을 포함한다. 시간 체크는 cooperative 방식이므로 요청 ms와 출력/스케줄링 지연을 구분한다.

| 모델 | 요청 | 실제 응답 ms |
|---|---:|---:|
| ZERO 초기 | 100ms | 101.015 |
| ZERO 초기 | 1000ms | 1001.235 |
| ZERO 초기 | 10000ms | 10001.859 |
| Lichess 10,000 | 100ms | 100.948 |
| Lichess 10,000 | 1000ms | 1001.231 |
| Lichess 10,000 | 10000ms | 10001.732 |

두 모델의 stop 지연은 각각 약 0.523ms / 0.622ms였다. 무한 탐색 중 quit도 정상 종료했다.

### 메모리 회수

1MiB 트리 예산, MaxNodes 16, 1,024회 완료 시뮬레이션으로 회수를 강제했다. 초기 모델은 125회 회수 / 1,014 노드 퇴거, 후보는 115회 회수 / 1,009 노드 퇴거를 기록했다. 예약 트리 메모리는 각각 33,088 bytes였다. 보존된 root Visits/Q/prior가 모든 회수 전후 일치해야 통과하도록 검사했다. 프로세스 전체 RAM을 이 수치로 표시하지 않는다.

## Windows 전술 교차 검증

같은 100문제, 100ms, seed 1, 검색 스레드 1, 동일한 Windows 하드웨어/CPU 집합에서 셀마다 3회 실행했다. 모든 셀의 파일 해시·실제 로딩 텐서 해시·엔진 해시·옵션·응수를 [metadata](reports/windows-native/40146054219-1/matrix/metadata.json), [CSV](reports/windows-native/40146054219-1/matrix/summary.csv), 셀별 `.json.gz`에 저장했다. 총 1,800회 응수이며 고유 문제 수는 100개다.

| 모델 | 기본 MCTS | 전술 검증 | OVERLORD 조합 |
|---|---|---|---|
| ZERO 초기 챔피언 | 22 / 22 / 22 | 45 / 45 / 45 | 50 / 49 / 50 |
| Lichess 10,000개 후보 | 34 / 35 / 34 | 48 / 48 / 48 | 47 / 47 / 47 |

**56/100은 이 Windows 러너에서 재현되지 않았다.** 초기 모델 조합은 49~50/100이다. Linux 원래 바이너리에서는 56을 재관측했고, 새 digest 계측 바이너리의 Linux 3회는 56/56/56이었다. Windows의 결과를 Linux 성과로 교체하거나 두 플랫폼을 합산하지 않는다.
- 같은 Windows 모델에서 탐색만 바꾼 중앙값 차이: ZERO 전술 +23 / 조합 +28, Lichess 후보 전술 +14 / 조합 +13문제.
- 같은 Windows 탐색에서 모델만 바꾼 중앙값 차이: 기본 +12 / 전술 +3 / 조합 −3문제. 학습 효과와 탐색 효과에 상호작용이 있어 단순 합산하지 않는다.
- 초기 모델 조합에서 proof 또는 전체 후보 비교가 끝난 문제는 Linux 78/100, Windows 69~70/100이었다. 더 적은 완료 검증이 관측됐지만 하드웨어/컴파일러도 다르므로 이것만으로 차이의 인과관계를 확정하지 않는다.
- 표본은 고정된 100개 공식 퍼즐의 게시된 첫 응수다. 다른 승리 수도 존재할 수 있으며 3회 반복을 300개 독립 표본으로 취급하지 않는다. Elo 상승이나 모델 승격을 결정하지 않았다.

## 실패 및 결과 전달 기록

- 첫 [워크플로 실행](https://github.com/Blucy1004/Chromatic/actions/runs/40145840154)은 job 생성 전에 실패했다. job-level env의 경로 문맥을 수정한 뒤 실제 Windows 실행이 성공했다.
- 실제 네이티브 결과는 모두 성공했으나 최초 문서 생성기는 상대 결과 경로와 절대 문서 경로를 비교하다 실패했다. 그 뒤 PowerShell이 후속 Git 명령의 성공으로 오류를 가려 결과 커밋도 누락됐다. 상대 경로 처리를 고치고 중간 명령 exit code 검사를 추가했으며 [문서 생성기 회귀 테스트](reports/provenance/windows-report-unit.log)가 통과했다.
- 원본 산출물에는 run-summary/build/CTest/protocol/18개 matrix 셀이 남았다. [복구 실행](https://github.com/Blucy1004/Chromatic/actions/runs/40147294534)은 그 **같은 Windows 산출물**의 OS·컴파일러·바이너리 SHA·x64 PE를 확인해 Git으로 보존했다. 복구 작업의 Ubuntu 실행을 Windows 엔진 실행으로 계산하지 않는다.
- 원본 ZIP SHA-256: `129581eac8a9de223df5eebd8286dcb0014e289c98d244431ba30b121d2ae97d`. Git으로 가져온 원시 파일 31개의 바이트가 ZIP과 정확히 일치한다. [최종 감사 검증](reports/provenance/final-audit-validation.json)에 기록했다.
- 원본/미러 PT와 ZERO, 기존 배포 실행 파일 70개에 대한 해시 보존을 확인했다. 새 기능은 기본 비활성, Threads 기본값은 1이며 후보 모델을 승격하지 않았다.

## 다시 실행

`.github/workflows/overlord-windows.yml` 또는 Visual Studio 2022 x64 개발 환경에서 CMake Release → CTest → `tools/platform_validation.py --require-windows` → `tools/model_matrix.py --require-windows --repeats 3` 순서로 실행한다. 비교 조건과 파일 경로는 각 metadata.json에 있다. 모델 계보 전체는 [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md)를 참조한다. 다른 Windows 버전/CPU 또는 배포 환경의 MSVC 재배포 런타임까지 검증한 것으로 확대하지 않는다.
