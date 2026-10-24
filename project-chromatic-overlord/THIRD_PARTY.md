# 외부 구성요소

* PyTorch: 모델 학습에만 사용합니다. [공식 저장소 및 라이선스](https://github.com/pytorch/pytorch).
* NumPy: 학습 데이터 인코딩과 검증에 사용합니다. [공식 저장소](https://github.com/numpy/numpy).
* python-chess: Python 도구의 규칙 대조, 대국 관리, 외부 교사 UCI 통신에 사용합니다. GPL-3.0-or-later. [공식 저장소](https://github.com/niklasf/python-chess).
* Stockfish 19: 제한된 128개 위치의 교사 데이터 생성에 외부 프로세스로 사용했습니다. GPL-3.0. [해당 릴리스/소스](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19). 실행파일·소스는 이 프로젝트의 배포 ZIP에 포함하지 않습니다. 모델/데이터의 교사 SHA-256은 `runs/bootstrap/teacher.meta.json`에 기록되어 있습니다.

C++ 엔진의 기물 이동, 탐색, 모델 추론 코드는 이 프로젝트에서 독립적으로 작성했습니다. 기존 무영 엔진의 소스를 사용하거나 변경하지 않았습니다.

이 저장소는 GPL-3.0-or-later로 제공합니다. 라이선스 본문은 `LICENSE`를 확인하세요.
