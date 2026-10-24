# 외부 구성요소와 상속한 모델

이 별도 학습 도구는 GPL-3.0-or-later로 제공한다. [LICENSE](LICENSE)를 확인한다. 기존 CHROMATIC의 모델 구조·인코딩·체크포인트·내보내기 코드는 `vendor/zero`에 복사해 재사용하며 원래 엔진 소스는 수정하지 않았다. Muyeong 코드는 사용하거나 수정하지 않았다.

실행 의존성은 [requirements.txt](requirements.txt)에 고정했다. 각 패키지 자체의 라이선스는 해당 배포본과 공식 저장소를 따른다.

- [PyTorch](https://github.com/pytorch/pytorch): CPU 신경망 및 Optimizer, 체크포인트 상태.
- [NumPy](https://github.com/numpy/numpy): 인코딩과 난수 상태.
- [python-chess](https://github.com/niklasf/python-chess): FEN·합법수·PV·UCI·게임 결과. GPL-3.0-or-later.
- [python-zstandard](https://github.com/indygreg/python-zstandard): bounded Zstandard 프레임 처리.
- [psutil](https://github.com/giampaolo/psutil): 프로세스 메모리 및 CPU affinity.
- [pytest](https://github.com/pytest-dev/pytest): 개발 검증.

신규 학습 데이터는 공식 Lichess의 CC0 Evaluation Database와 CC0 오프닝 목록을 사용한다. 보정 및 독립 평가용 공식 게임·퍼즐 출처와 라이선스 근거는 [docs/SOURCES.md](docs/SOURCES.md)에 기록했다. 데이터의 CC0와 프로그램의 GPL은 구분한다.

`champion/model.pt`와 `champion/default.zero`는 기존 CHROMATIC 챔피언의 바이트 그대로 복사한 시작점이다. 원래 CHROMATIC의 기록상 챔피언은 외부 Stockfish 교사의 제한된 부트스트랩에서 시작했다. 이 새 학습 도구 구현·검증에서는 Stockfish 실행파일을 호출하거나 배포하지 않았다. 상속한 챔피언의 과거 데이터 전체를 CC0였다고 주장하지 않는다. 보존한 원본 프로젝트의 `THIRD_PARTY.md`와 `runs/bootstrap/teacher.meta.json`에 기존 이력이 남아 있다.

비교에 사용한 메모리 회수 개선 엔진은 기존 경로에서 읽어 실행했으며 이 학습 도구 ZIP에 엔진 바이너리는 넣지 않는다. 독립 평가 도구에는 사용자가 보유한 CHROMATIC 엔진 경로를 `--engine`으로 지정한다.
