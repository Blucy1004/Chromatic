# 정확성 감사

## 기준선과 범위

원본 소스·테스트·문서·모델·대국 기록을 조사하고 SHA-256 inventory를 만들었습니다. v0.1 ZIP과 실행파일·모델·원본 소스는 `baseline/`에 보존했습니다. 별도 `overdrive/v1` 개발 경로에서 변경했으며 원래 PROJECT ZERO와 무영은 변경하지 않았습니다. 원본 보존의 최종 해시 대조는 `reports/preservation-final.json`에 있습니다.

기준선 테스트를 재실행했고, 128개 교사 위치의 200스텝 학습 가중치가 원래 모델과 정확히 같았습니다. 기준 자가대국도 4판 중 3판 완료·1판 중단·190개 위치로 재현됐습니다. 결과는 `reports/baseline-reconstruction.json`입니다.

## 확인하고 고친 결함

| 항목 | 확인한 문제 | 조치와 회귀 근거 |
|---|---|---|
| 비정상 활성값 | 유한하지만 매우 큰 가중치가 overflow를 만들면 prior NaN과 잘못된 선택으로 이어질 수 있음 | 층 출력의 유한성 검사, UCI 오류/완료 응답, overflow 회귀 테스트 |
| 모델 magic | 잘린 헤더를 검사하기 전에 버퍼를 문자열로 비교 | 버퍼 초기화와 read 성공 확인, 버전·크기·EOF 검사 |
| FEN | 카운터 상한·불가능한 캐슬링 권리·백랭크 폰 등의 검증 부족 | 위치/권리/카운터 검증과 트랜잭션식 position 갱신 |
| 잘못된 go | 오류 뒤 bestmove 완료가 없어 클라이언트가 기다릴 수 있음 | error 뒤 bestmove 0000, 다음 명령 응답 검사 |
| Python UCI 동기화 | error 뒤 남은 readyok가 다음 요청의 응답으로 소비될 수 있음 | 준비/대국 응답을 끝까지 받은 뒤 오류를 전달하도록 변경 |
| 입력 모델 없음 | 시작 시 UCI를 열기 전에 종료 | 진단을 알리고 UCI 응답을 유지, 새 ModelFile 설정 가능 |
| optimizer NaN | 저장 경로에서 optimizer 수치 검증 부족 | 저장 전 검사로 마지막 정상 체크포인트 보존 |
| 반복 오프닝 통계 | 같은 오프닝의 반복을 독립 표본으로 세면 신뢰도가 부풀 수 있음 | 오프닝 cluster 신뢰구간과 중복 제외 pair-sign SPRT, 반복 200판 fixture에서 승격 거부 |
| 미완료 대국 | 결과를 모르는 정책 데이터가 충분히 내구적으로 보존되지 않음 | pending trace와 주기적 원자 저장·수 제한 연장/재개 |
| 빌드 의존성 | 한국어 컴파일러의 잘못된 헤더 의존성 prefix로 이전 구조체 ABI가 섞임 | UTF-8 코드페이지와 기본 전체 재빌드, 새 빌드/회귀 검증 |

## PUCT 감사

기존 부호와 WDL 관점은 일관됐습니다. 이를 결함으로 꾸며서 변경하지 않았습니다. 새 트리에서도 부모 관점 Q, 방문 합, WDL W-L, 간선별 부호 반전, terminal draw/mate와 legal masking을 유지했습니다. 스칼라·증명 비활성 모드에서 20개 위치의 수 선택/방문 분포/가치가 기준선과 정확히 같았습니다.

이제 자식이 자신의 승리를 확신하는 인공 WDL 모델을 사용하여 루트가 손실로 받아들이는지 직접 검사합니다. **역전파의 마이너스 하나만 제거한 변이 실행파일은 이 테스트에서 실패했습니다.** `reports/sign-mutation.json`에 실제 종료 코드와 오류가 있습니다. 메이트 보호가 이 검사를 가리지 않도록 해당 테스트는 증명 모듈을 끕니다.

루트 방문 합, 정규화된 prior, noise/temperature의 seed 재현성, 64KiB 트리 제한에서 진행·메모리 상한을 검사했습니다. 현재 트리 재사용과 transposition은 없으므로 반복/50수 이력이 다른 위치를 잘못 합치는 경로를 추가하지 않았습니다.

## 실행한 게이트

* C++ 컴파일과 핵심 테스트, 5개 perft 위치: startpos d5=4,865,609, Kiwipete d4=4,085,603, 추가 674,624 / 422,333 / 2,103,487.
* 독립 python-chess와 1,008개 위치의 합법 수/FEN/features/action/종료 대조.
* UCI nodes/time/infinite+stop/isready, 0초 시계, 불법 입력과 종료 위치.
* 전체/선택 정책 헤드의 동일 행 일치, Scalar/SIMD/PyTorch 대조.
* 모델·optimizer·generator·RNG의 연속/중단 학습 동등성, 이전 checkpoint에서 되돌아온 재학습, 데이터 변경 거부.
* Replay 용량/중복/검증 누수 방지, 미완료 기록의 보존·재개, 통제된 한 위치 과적합, AdamW/SGD 경로.
* PGN·쌍 대결 재시작·불완료 처리·통계 의사결정·SHA 확인 rollback.
* AddressSanitizer 빌드에서 핵심/perft 및 4개 안전성 테스트.

ASAN은 실행된 경로에서 메모리 오류를 보고하지 않았습니다. 모든 가능한 입력의 무결성을 증명한 것은 아닙니다. 이 MSVC 환경에서는 ThreadSanitizer를 실행하지 않았으며, 새 공유 트리 병렬 MCTS도 도입하지 않았습니다. [MSVC AddressSanitizer 공식 문서](https://learn.microsoft.com/en-us/cpp/sanitizers/asan?view=msvc-170)를 기준으로 별도 검사 빌드를 구성했습니다.

최종 자동 게이트 요약은 `reports/release-gates.json`이고, 개별 JSON과 ASAN 로그가 함께 제공됩니다. 미실행 검사는 통과라고 표시하지 않습니다.
