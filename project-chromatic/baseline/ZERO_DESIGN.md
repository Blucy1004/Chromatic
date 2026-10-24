# PROJECT ZERO 기술 명세

## 구성과 경계

```text
선택적 교사 UCI -> 제한된 bootstrap JSONL -> PyTorch 초기 모델
                                                |
                                      FP32 .zero export
                                                |
                  C++ 합법 수 + PUCT + WDL <- 챔피언 가중치
                              |
                    자가대국 방문 수 / 실제 결과
                              |
                최근 세대 replay -> PyTorch 후보 학습
                              |
              색 교환 후보 대결 -> 통계적 승격 판정
```

`src/board.cpp`는 독립 규칙 구현, `src/search.cpp`는 PUCT, `src/network.cpp`는 표준 라이브러리만 사용하는 추론입니다. Python의 `chess`는 교사 통신, 대국 관리, 독립 검증과 데이터 재구성에 사용합니다. C++은 Python 규칙 엔진이나 외부 엔진을 호출하지 않습니다.

## 체스 상태

64칸 mailbox, a1=0/h8=63. 백 기물은 +1..+6, 흑은 -1..-6이며 P/N/B/R/Q/K 순서입니다. 위치에는 둘 차례, 4비트 캐슬링 권리, 원시 앙파상 칸, 반수/전체 수 카운터, 반복 위치 해시 이력을 보존합니다.

의사 합법 수를 만든 뒤 이동한 위치의 자기 왕 공격 여부로 필터링합니다. 캐슬링은 왕과 룩의 위치, 빈 칸, 출발/통과/도착 칸 공격 여부를 확인합니다. 앙파상은 제거되는 폰을 반영한 왕 안전성 검사로 discovered check를 처리합니다. 승격은 N/B/R/Q 모두 생성합니다.

반복 해시는 기물·차례·캐슬링·**실제로 합법적인 앙파상 수가 있을 때만 EP 칸**을 포함합니다. 반수/전체 수 카운터는 반복 동등성에 포함하지 않습니다. 폰 이동/잡기 후에는 이전 반복 이력을 버립니다. C++과 Python 모두 이미 성립한 3회 반복과 100 halfmoves에서 즉시 무승부를 청구하는 대국 규칙을 사용합니다. 다음 수를 두면 청구할 수 있다는 사전 청구는 구현하지 않습니다. 메이트는 50수 판정보다 먼저 처리합니다.

## 입력 인코딩 v1

절대 좌표를 사용하며 흑 차례라고 보드를 회전하거나 색을 뒤집지 않습니다. FP32 20×64를 평탄화한 1,280차원입니다.

| 평면 | 의미 |
|---|---|
| 0..5 | 백 P,N,B,R,Q,K |
| 6..11 | 흑 P,N,B,R,Q,K |
| 12 | 백 차례면 전체 1, 흑이면 전체 0 |
| 13..16 | 백 O-O, 백 O-O-O, 흑 O-O, 흑 O-O-O 권리 |
| 17 | 원시 EP 칸 one-hot, 없으면 0 |
| 18 | min(halfmove,100)/100 전체 채움 |
| 19 | min(현재 위치 반복 횟수,3)/3 전체 채움 |

이력은 완전한 위치 복구와 반복 판정에 사용하지만 신경망에는 과거 기물 평면 전체를 제공하지 않습니다. 보드만 같은 서로 다른 이력의 장기 가치는 이 소형 인코딩으로 완전히 구분되지 않을 수 있습니다.

## 정책 인덱스 v1

출력은 4,672개 logits입니다.

* 보통 이동과 Q 승격: `from*64 + to`, 0..4095.
* N/B/R 승격: `4096 + from*9 + (to_file-from_file+1)*3 + (promotion_piece-2)`.
* `promotion_piece`는 N=2, B=3, R=4.

Q 승격과 일반 이동은 동일 인덱스를 쓰지만 같은 위치에서 같은 시작/도착 칸의 Q 승격과 비승격이 동시에 합법일 수 없으므로 합법 수 사이 충돌이 없습니다. N/B/R 승격은 전진·좌측 잡기·우측 잡기를 각각 구분합니다. 흑 승격도 절대 좌표 공식이 같습니다.

불가능한 이동 logits는 탐색 softmax와 학습 log_softmax 전에 제외합니다. 부트스트랩과 자가대국 JSONL은 `[action,probability]`의 희소 분포로 저장합니다.

## Policy/WDL 모델

기본 구조: `Linear(1280,64) -> ReLU -> 2×ReLU(x+Linear(x))` 뒤에 `Linear(64,4672)` Policy와 `Linear(64,3)` WDL 헤드를 둡니다. WDL 순서는 **현재 둘 차례의 Win, Draw, Loss**입니다. 가치값은 `W-L`입니다. 폭 1..1024, residual 층 0..16을 export 형식에서 지원합니다.

학습은 masked policy cross entropy + WDL cross entropy, AdamW(weight decay=1e-4), gradient norm clip=1, 학습률 `initial_lr*0.999^step`입니다. 데이터셋은 매 스텝 전용 CPU Torch generator로 복원 추출합니다. CPU 검증은 1개 Torch 스레드와 결정적 연산을 사용합니다. CUDA 경로는 코드상 지원하지만 이번에 검증하지 않았습니다.

## C++ 모델 파일

Little endian FP32 전용 바이너리:

1. 8바이트 ASCII `ZERO0001`.
2. uint32 4개: inputs=1280, width, residual_blocks, actions=4672.
3. input, residual[0..N), policy, WDL 순서로 각 Linear 층의 row-major weight `[out,in]`, bias `[out]`.

레이어별 입력/출력 크기는 헤더로 결정합니다. 파일 로딩은 임시 Network에 수행하고 모든 데이터가 정상일 때 교체합니다. 차원 범위, EOF, 유한한 가중치, 불필요한 끝 바이트를 검사합니다. PyTorch 그래프의 FP32 연산 순서 차이를 고려한 절대오차 허용치는 2e-5입니다. Python `verify`는 입력, 합법 수/action, 모든 정책 logit, WDL 확률을 비교합니다.

## PUCT

간선은 부모 관점의 방문 수 N, 가치 합 W, prior P를 가집니다. 선택식은 `Q + cpuct*P*sqrt(parent_visits+1)/(1+N)`이며 방문하지 않은 간선의 Q는 0입니다. 잎 WDL을 현재 차례 관점으로 구한 후, 간선 하나를 올라갈 때마다 부호를 반전합니다. 종료 위치는 메이트=-1, 무승부=0입니다.

한 시뮬레이션마다 root 간선 방문 수가 정확히 하나 증가합니다. 자가대국 정책 타깃은 이 root 방문 수의 정규화입니다. root prior에는 Dirichlet(alpha=0.3), 기본 epsilon=0.25를 섞습니다. 초기 30플라이는 방문 수를 온도 1로 샘플링하고 그 뒤에는 최대 방문 수를 선택합니다. 대결은 noise=0, temperature=0입니다.

트리마다 노드 상한을 적용합니다. 상한 도달 후 새 잎은 평가하지만 트리에 저장하지 않습니다. 반복 판정이 경로에 의존하므로 현재는 transposition 병합을 사용하지 않습니다. 원자적 stop과 시뮬레이션 사이의 시간 검사로 UCI 중지를 처리합니다.

## 데이터와 복구

JSONL record: `schema`, `source`, `root_fen`, 시작 위치부터의 `moves`, 현재 `fen`, sparse `policy`, `wdl` + 출처 메타데이터. FEN만 저장해서 반복 이력을 잃지 않도록 전체 이동 이력을 보존합니다. 로더는 이력/FEN, 합법 action, 중복, 비음수/유한 확률과 분포 합을 검사합니다.

자가대국은 대국 단위로 JSONL과 SHA-256 메타데이터를 원자적으로 저장합니다. 완료된 대국만 WDL 라벨을 받으며, 잘린 대국의 JSONL은 비어 있습니다. 학습 checkpoint에는 모델/optimizer/진행 스텝/RNG/generator/데이터 SHA/설정을 저장합니다. 저장이 끊겨 남은 `.tmp`는 복구 대상으로 읽지 않습니다.

세대 상태는 완료한 세대 수와 챔피언 checkpoint 경로를 저장합니다. 후보 승격 결정 후에만 세대 상태를 원자적으로 갱신합니다. 대결 도중 중단되면 대결 전체를 재실행합니다. 학습 도중에는 마지막 정상 checkpoint로 복구합니다.

## 확장 우선순위

현재 병목은 소형 MLP의 표현력과 단일 위치 CPU 추론입니다. 강도 개발에는 충분한 전술/엔드게임 홀드아웃, 다양한 오프닝과 결과, CNN/ResNet, 여러 자가대국 작업자와 GPU 배치 추론이 필요합니다. 이후 정책 형식을 유지하거나 버전을 올려 백엔드를 교체할 수 있습니다. 트리 재사용·가상 손실·transposition은 반복 이력과 50수 상태가 다른 경로를 잘못 합치지 않도록 별도 테스트와 함께 도입해야 합니다.

참고: [PyTorch state_dict와 저장](https://docs.pytorch.org/docs/stable/notes/serialization), [Stockfish 공식 배포](https://github.com/official-stockfish/Stockfish/releases), [python-chess 문서](https://python-chess.readthedocs.io/en/latest/).
