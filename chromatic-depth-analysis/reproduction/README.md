# 재현

이 폴더는 실행에 사용한 작업 스크립트의 보존 사본이다.

- C++ 엔진과 probe는 `project-chromatic-depth/README.md`의 명령만으로 독립 실행할 수 있다. 실행 파일 옆의 `default.zero`가 동일 모델이다.
- 그래프 재생성은 `python depth_analysis.py --input ../runs.jsonl --out ..`로 가능하다. 상대 경로는 이 폴더에서 실행할 때 기준이다. requirements.txt는 당시 분석 패키지 버전이다.
- 전체 실험/배포 동등성 스크립트는 원래 작업공간의 `work/` 배치를 전제로 한다. 사본을 작업공간 `work/`에 두고 실행한다. 원본 `outputs/project-chromatic`과 기존 원인 분석 자료를 읽어 보존 여부도 검증한다.
- 각 `raw/*.request.json`에는 정확한 명령 인자와 조건이, 같은 이름의 stdout 파일에는 가공 전 probe JSON이 있다. 다른 경로에서 probe를 재실행할 때 executable/model 경로만 새 위치로 바꾼다.
- 시간 모드는 기계 상태에 따라 완료 방문수가 달라진다. 동일 N 모드의 탐색 통계와 벽시계 시간 결과는 구분한다. 과거 N 고정 대조는 별도 historical_anchor 그룹이다.
