> 아래 공식 표본의 수집·프레임 검증 증거는 보존한 이전 학습기에서 읽어 복사했다. ARC 구현 중 원격 학습 데이터 다운로드는 하지 않았다. 현재 ARC 성능·정확성 결과는 별도 PERFORMANCE.md와 numeric/recovery 보고서에 있다.

# 데이터 출처와 검증 범위

이 문서는 2026-10-24에 실제 수집한 표본과 저장된 메타데이터를 설명한다. 새 학습·보정·검증 자료는 Lichess가 공식 배포한 CC0 데이터에서 가져왔다. 데이터 라이선스의 근거는 [공식 데이터베이스 안내](https://database.lichess.org/)와 [평가 데이터 안내](https://database.lichess.org/#evals), 오프닝 저장소의 고정 커밋에 있는 [CC0 원문](https://raw.githubusercontent.com/lichess-org/chess-openings/a6189a30dc273ccb21fc2536a9a2fefd5592a67a/COPYING.txt)이다. 로컬 표본의 라이선스·원본 URL·버전·처리 범위·해시는 `fixtures/*.source.json`과 `fixtures/official-sample-manifest.json`에 기록했다.

| 자료와 용도 | 공식 원본 | 고정한 버전 |
|---|---|---|
| 평가 JSONL/Zstandard: Policy 및 보정된 WDL 교사 라벨 | [Lichess Evaluation Database](https://database.lichess.org/lichess_db_eval.jsonl.zst) | ETag `"6ac01121-5388faeef"`, Last-Modified `2026-10-02 20:16:33 GMT` |
| 2026년 9월 대국 PGN/Zstandard: 실제 대국 결과와 평가 주석을 이용한 독립 보정·검증 | [공식 Standard rated PGN](https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst) | ETag `"6ac08d73-6cdeaccb7"`, Last-Modified `2026-10-03 05:06:59 GMT` |
| 오프닝 TSV: 시작 위치 및 초기 합법 수열의 학습 앵커 | [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings) | 커밋 `a6189a30dc273ccb21fc2536a9a2fefd5592a67a` |
| 퍼즐 CSV/Zstandard: 학습에서 제외한 독립 전술 테스트 | [공식 퍼즐 데이터베이스](https://database.lichess.org/lichess_db_puzzle.csv.zst) | ETag `"6abf70a1-125007eb"`, Last-Modified `2026-10-02 08:51:45 GMT` |

ETag·Last-Modified·원본 크기는 HTTP 객체의 변경을 감지하기 위한 버전 식별자다. 별도로 기록한 SHA-256은 실제 읽은 압축 접두부 또는 저장된 로컬 파일의 바이트를 식별한다. 원본 전체를 읽지 않았으므로 거대한 원본 파일 전체의 SHA-256을 계산했다고 주장하지 않는다. 평가 안내 페이지의 갱신일 `2026-10-05`와 HTTP Last-Modified는 각각 보존했다.

아래 바이트 수는 해당 표본 수집 또는 프레임 재검증 실행에서 실제 읽은 양이다. 초기 수집, 재검증, 소량 HTTP 조사 전체의 누적 네트워크 사용량을 뜻하지 않는다. `MiB`는 1,048,576바이트다.

| 자료 | 보존한 표본 | 해당 실행의 압축 읽기량 | 해당 실행의 압축 해제량 | 원본 프레임 검증 범위 |
|---|---:|---:|---:|---|
| 평가 데이터 | 첫 60,000행 | 11,506,841바이트 | 67,108,864바이트, 64MiB | 소비한 표준 프레임 2개를 디코더로 끝까지 검증, 부분 프레임 0개. 건너뛰기 프레임 2개 처리 |
| 대국 데이터 | 첫 5,000개 완결 PGN 레코드 | 재검증 4,692,881바이트 | 재검증 33,554,432바이트, 32MiB | 소비한 표준 프레임 1개의 내장 체크섬 검증, 부분 프레임 0개. 건너뛰기 프레임 1개 처리 |
| 퍼즐 | 첫 100개 레코드 | 65,536바이트 | 65,536바이트 | 의도적으로 중단한 압축 접두부다. 원본 프레임 전체의 체크섬은 확인하지 못함 |
| 오프닝 | `a.tsv`부터 `e.tsv` 및 `COPYING.txt` 전체 | 압축하지 않은 파일 합계 402,112바이트 | 해당 없음 | 각 파일 전체 SHA-256 기록 |

평가 원본의 확인된 전체 크기는 22,423,777,007바이트, 대국 원본은 29,224,520,887바이트, 퍼즐 원본은 307,234,795바이트다. 이 파일들을 통째로 내려받지 않았다. 원본의 일부 프레임만 검증하고 그 프레임에 포함된 필요한 행까지 사용하는 방식으로 소규모 검증을 수행했다. 소비한 프레임의 정상 여부와 아직 읽지 않은 나머지 원본의 정상 여부는 별개의 범위다.

평가 표본은 원본 프레임 검증 후 로컬에서 체크섬을 켠 Zstandard로 다시 압축했다. 선택한 60,000개 JSONL 행의 SHA-256은 `cbf0a49b304384e3675b64f07bcb4ee04b80bf9d9b31230497b1bd1e2720e685`이며, 검증 전후의 선택 행과 로컬 압축 파일 해시가 일치했다. [원본 프레임 검증 보고서](../reports/official-frame-verification.json)에 원본 식별자, 압축 오프셋, 행 커서, 압축 접두부 해시를 남겼다. 첫 프레임의 8MiB 윈도·체크섬 플래그·미지정 content size 및 다음 프레임의 헤더에 대한 소량 관측은 [프레임 조사 기록](../reports/frame-probe.json)에 있다.

대국 표본의 초기 수집은 압축 1,703,936바이트, 압축 해제 11,796,480바이트에서 멈췄다. 후속 검증은 동일한 ETag와 Last-Modified를 확인하고 바이트 0부터 다시 시작해 첫 원본 프레임을 완전히 확인했다. 해당 프레임은 8MiB 윈도와 내장 체크섬이 있고, 압축 해제된 크기는 32MiB다. 처음 저장한 5,000게임의 PGN 전체 SHA-256, 순서가 있는 게임 ID 5,000개, 평가 주석 31,204개가 모두 일치했다. PGN 본문, 추출된 보정 행, 보정 파라미터 파일은 변경하지 않았다. [대국 프레임 검증 보고서](../reports/game-frame-verification.json)와 `official-games-sample.pgn.source.json`의 `frame_verification`에 새 증거를 추가했다. 보정 행 내부의 `source`는 생성 당시의 출처 기록으로 유지했다.

원본 프레임 검증은 Zstandard 디코더의 프레임 구조 및 프레임 내장 체크섬 검사를 뜻한다. 체크섬이 없는 프레임에서는 구조를 읽고 해제할 수 있어도 같은 수준의 전송 손상 검출을 보장할 수 없다. `verified_frames_with_checksum`, `verified_frames_without_checksum`을 구분하며, 최초 4개 검증 프레임의 헤더만 저장해 관측 기록의 메모리 사용을 제한한다. `verified_complete_zstd_frames`라는 상태 문자열만으로 모든 프레임에 체크섬이 있었다고 해석하면 안 된다. 이번 대국 재검증은 체크섬이 있는 프레임 1개, 없는 프레임 0개였다.

퍼즐 표본은 100개만 수집하고 추가 대규모 다운로드를 하지 않았다. 로컬 CSV의 SHA-256과 첫 상대 수·정답 응수의 합법성을 확인했으며, 원본 압축 프레임 전체 체크섬의 미확인 상태는 [퍼즐 출처 메타데이터](../fixtures/official-puzzles.csv.source.json)에 그대로 남겼다. 로컬 해시가 일치한다는 사실은 저장 이후 바이트의 동일성을 확인하는 근거다. 원본에서 중단한 프레임 전체에 대한 검증 근거로 확장하지 않는다. 퍼즐은 독립 전술 테스트용이고 해당 가족은 신경망 학습에서 제외한다. 공개된 첫 응수와의 일치율은 다른 승리 수의 가능성을 포함한 제한된 지표다.

로컬 파일의 SHA-256은 다음과 같다. 파일 이름은 모두 `fixtures/` 기준이다.

| 파일 | SHA-256 |
|---|---|
| `official-evaluations.jsonl.zst` | `feb08110ffcef588f05c30bb7ca93f5e19897e85c3d84678aff076f51dd84e2b` |
| `official-games-sample.pgn` | `867b3917b68692f73bc6b4d5d88f655373b682c5ca2be236b3ea1bd71efd3bf7` |
| `official-calibration-rows.jsonl` | `9764845735ca8f992389c8c1ec59ac0da1e2a5ec273ed1858875ccb6777fe210` |
| `calibration.json` | `82577601c9e455e9c3b40125260df9bf1541db1cff473fe49702765bc3816798` |
| `official-puzzles.csv` | `c46a5c5d83a27e4418a4a13aa196b4097c9aa863d76ce18386d61c8e628cecfb` |
| `official-openings-a.tsv` | `3282e4c9155289a29224f9a85fba0decb46efa35c2fa5e39662d2ac48fb0f793` |
| `official-openings-b.tsv` | `1d5ed134ebbd87915ead5683416e37583bb3e5ae3582b4d8cb5cb4aa7ef4f623` |
| `official-openings-c.tsv` | `45e843c7b7e79a0d04d5bda879c9bd23f5a10974488bd8615e4f4c5d0749986f` |
| `official-openings-d.tsv` | `1cf4057a8aece94f27bb3f278ad410ee2844831a642364eeceaafa091dbcceda` |
| `official-openings-e.tsv` | `48c6f6645b031a5937c8386839debab0e2e48837251687ba1afb6369677f8546` |
| `official-openings-COPYING.txt` | `a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499` |

오프닝 파일은 각각 67,257 / 78,124 / 132,304 / 73,318 / 44,061바이트이고 CC0 원문은 7,048바이트다. 고정 커밋별 원본 URL과 응답 ETag는 [오프닝 출처 메타데이터](../fixtures/official-openings.source.json)에 있다. 오프닝은 관측된 수열의 약한 Policy 지도에 사용한다. 그 수열만으로 승패가 확인된 것으로 취급하지 않으며 해당 Value 가중치는 0이다. 실제 대국의 결과와 평가 주석에서 추출한 보정 자료는 483게임·3,831행이며, 보정 fit/validation/test는 게임 ID로 분리한다. 신경망의 실제 결과 보정 평가는 독립 test 92게임·735행을 사용하고 각 게임의 총 가중치를 1로 한다. 인간 대국의 결과와 완벽한 플레이의 이론적 WDL은 구분한다.

HTTP Range 지원은 광고 헤더만 보고 판단하지 않았다. 평가 원본에 `Range: bytes=0-1023`과 `Range: bytes=1024-2047`을 보내 각각 206 및 정확한 `Content-Range`를 확인했다. [HTTP 조사 원문](../reports/http-probe.json)에 응답 헤더와 읽은 1,024바이트씩의 해시가 있다. 대국과 퍼즐의 초기 수집도 각각 206 응답 및 요청 범위의 `Content-Range`를 기록했다. 서버 동작은 달라질 수 있으므로 실제 재접속 시에도 객체 식별자와 응답 범위를 다시 검사한다.

진행 중인 프로세스가 네트워크 오류를 만나면 살아 있는 디코더 상태를 유지한 채 정확히 이어지는 압축 바이트를 Range로 요청한다. 서버가 Range를 무시하고 200으로 응답하면 이미 읽은 압축 접두부를 버린 뒤 디코더에 중복 입력 없이 이어 붙인다. 재시도 횟수는 제한한다. 임의의 압축 중간 바이트를 독립적인 Zstandard 시작점으로 해석하는 방식은 사용하지 않는다. 서버의 Range 지원은 압축 해제 상태를 재구성해 주지 않는다.

새 프로세스에서 복구할 때는 압축 바이트 0부터 재생하고, 저장한 물리적 행 수와 접두부 SHA-256을 확인한 뒤 다음 행을 제공한다. `compressed_offset_diagnostic`는 진단용이다. 초기 위치부터 다시 읽는 복구 비용은 처리한 접두부 길이에 비례하며, 늦은 중단일수록 다시 읽는 네트워크·압축 해제 비용이 커진다. 현재 구현은 임의 위치에서 즉시 시작하는 색인 복구를 제공하지 않는다. 앞으로 프레임 단위 색인을 추가하려면 검증된 프레임 경계와 프레임 사이에 걸친 행의 상태까지 보존해야 한다.

메모리와 디스크 사용은 압축 파일 전체 크기에 비례하지 않도록 제한한다. 디코더는 64KiB 단위로 출력하고, 한 프레임을 제한된 임시 디스크 spool에 쓴다. 프레임 검증을 마친 뒤 해당 행들을 학습 파이프라인에 노출한다. 따라서 소량의 행만 필요해도 그 행이 든 프레임 전체를 검증할 공간과 압축 해제 예산은 필요하다. 매우 큰 프레임이 상한을 넘으면 기본 모드가 중단한다. 전체 프레임을 메모리에 올리는 우회 처리는 없다.

| 소규모 검증 도구 | 압축 입력 상한 | 누적 압축 해제 상한 | 단일 프레임 spool 상한 | 추가 상한 |
|---|---:|---:|---:|---|
| `tools/verify_official_sample.py` | 16MiB | 128MiB | 128MiB | fixture 디렉터리 총 256MiB, 60,000행, 180초 |
| `tools/verify_game_sample.py` | 8MiB | 64MiB | 64MiB | fixture 디렉터리 총 256MiB, 5,000게임, 180초 |
| 초기 퍼즐 표본 수집 | 1MiB | 4MiB | 부분 프레임 수집이므로 전체 spool 검증 없음 | 100개 레코드 |

일반 `RecordStream`의 기본 상한은 압축 입력 256MiB, 누적 압축 해제 2GiB, 한 프레임 spool 128MiB, 한 행 1MiB, 선언 윈도 128MiB다. 실제 학습 파이프라인은 실행 설정의 다운로드·압축 해제 상한과 디스크·메모리 보호 조건을 별도로 전달하므로 실행 설정과 `resource-policy.json`도 확인해야 한다. 단일 프레임·행·윈도 제한은 서로 다른 항목이다. 프레임 헤더의 윈도 크기는 바이트 단위로 직접 검사한다. 전체 프레임 내장 체크섬 검증을 생략하는 `allow_partial_frames`는 명시적으로 요청하는 표본 조사용 기능이며, 기본값은 false다. 부분 사용 시에는 `partial_frame_checksum_unavailable` 상태를 기록하고 완전한 원본 프레임 검증의 근거로 사용하지 않는다.

프레임 spool은 정상 종료·오류·반복자 종료 시 삭제한다. 강제 프로세스 종료가 남긴 임시 파일은 데이터 디렉터리의 실행 잠금을 획득한 뒤 그 실행이 소유한 이름과 경로만 정리한다. 학습 데이터 디렉터리의 `ledger.sqlite3`에는 소스 식별자·프레임 통계·행 커서·처리 결정을 함께 기록한다. 각 학습 행은 원본 URL·버전·행 번호·소스 식별자 해시를 가진다. 로컬 fixture의 `.source.json`, 체크포인트 settings의 `source_provenance`, immutable bundle manifest와 함께 추적할 수 있다. 외부 파일에 단순히 CC0 문자열을 붙이는 것만으로 공식 출처가 입증되는 것은 아니므로, 기본 실행 경로는 공식 URL과 로컬 파일 해시를 검사한다.

`champion/model.pt`와 `champion/default.zero`는 기존 CHROMATIC 모델의 보존 복사본이다. 새 CC0 자료를 사용한 이번 추가 학습의 출처 기록과, 이미 존재하던 챔피언의 과거 전체 학습 데이터 이력은 구분한다. 이번 작업은 과거 이력을 역으로 복원하거나 그 자료 전체의 라이선스를 재확인하지 않았다. 보존된 모델의 SHA-256은 각각 `99151f0c57212edd6786e2bbbafc1717bb15bee9a24341e69c5606c478bb4b84`와 `f6a00902c23e7f32f96b028ac36464cc9012cd8774d5e6b0dee5e1c94e1c03c9`이다. 원래 소스·vendor 코드와 그 `LICENSE`의 조건도 CC0 데이터 라이선스와 별도로 유지한다. 새 데이터가 CC0라는 이유로 기존 엔진·모델·과거 자료를 모두 CC0로 다시 표시하지 않는다.

원본 프레임에 체크섬이 하나라도 없으면 프레임 경계·디코딩 완료만 확인할 수 있으므로 커서의 무결성 상태를 `complete_frames_without_full_checksum`으로 낮춘다. 프레임별 체크섬 유무 횟수는 스트림 통계에 기록하며, 뒤에 체크섬 있는 프레임이 나와도 앞부분의 미검증 상태를 되돌리지 않는다. 이 상태는 온라인 원본의 완전한 체크섬 검증 또는 승격 근거로 인정하지 않는다. 이번 평가·게임 샘플에서 소비한 원본 프레임에는 모두 체크섬이 있었다.

이번 검증의 상세 실행 기록은 [스트리밍 회귀 검사](../reports/streaming-validation.json), [ARC 수치 검증](../reports/numeric-xpu-fp32.json), [보정 결과](../reports/calibration-report.json), [표본 통합 출처](../fixtures/official-sample-manifest.json)에서 확인할 수 있다.
