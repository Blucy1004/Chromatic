# Windows x64 Release 실행 패키지

**[ZIP 다운로드](https://github.com/Blucy1004/Chromatic/raw/e69e7383de7d5784fbe95fdbf5a665d0d2a21602/project-chromatic-overlord/packages/windows-x64/40161264588-1/CHROMATIC-Windows-x64-Release.zip)** · [GitHub 파일 페이지](https://github.com/Blucy1004/Chromatic/blob/e69e7383de7d5784fbe95fdbf5a665d0d2a21602/project-chromatic-overlord/packages/windows-x64/40161264588-1/CHROMATIC-Windows-x64-Release.zip)

크기 1,870,719바이트. MSVC 19.44 / x64 / Release / 정적 런타임(`/MT`) 빌드다.
배포 ZIP의 직접 DLL 의존성은 `KERNEL32.dll`뿐이며, 실행에 Python이나 별도 Visual C++ Redistributable 설치가 필요하지 않다.
Windows 10 1903+ / Windows 11 x64용이며 실제 네이티브 검사는 Windows Server 2022 x64에서 수행했다.

## 실행

1. ZIP 전체를 한 폴더에 압축 해제한다.
2. 콘솔 실행은 `run-console.cmd`를 더블클릭한다.
3. 체스 GUI에서는 UCI 엔진으로 `chromatic.exe`를 등록한다. 옆의 `default.zero`를 자동으로 로딩한다.

체스판 GUI는 포함하지 않는다. 콘솔에서는 `uci`, `isready`, `position startpos`, `go movetime 1000`을 차례로 입력하고, `bestmove` 후 `quit`로 종료한다.
파일 무결성과 실제 모델 로딩은 `self-check.cmd`로 확인한다.

## 포함 파일과 기본값

- `chromatic.exe`, 보존된 초기 챔피언 `default.zero`.
- 콘솔·자체 확인·재빌드 스크립트와 사용법.
- GPL 라이선스, 정확한 대응 C++ 소스·테스트·빌드 설정.
- `MANIFEST.json`, `SHA256SUMS.txt`, 네이티브 검증 기록.

Threads=1, 실험 기능 C/A/F/T와 ResearchMetrics는 모두 OFF다. 원래 MateProofNodes=256을 유지한다.
기존 배포본과 챔피언은 변경하지 않았으며 후보 모델을 승격하지 않았다. Release는 빌드 구성이고 엔진은 Operation Overlord 실험 버전이다. Elo 개선은 미확정이다.

## 실제 Windows 검증

- MSVC 정적 런타임 빌드와 기존 CTest 11개 통과.
- 압축 해제한 패키지에 대한 네이티브 실행 검사 33개 통과.
- 공백·한글·작은따옴표가 있는 경로, 다른 작업 폴더 및 basename-only argv[0]에서 모델 로딩 확인.
- 실제 로딩 텐서 SHA-256, 100ms/1초/10초, go/stop/quit, 옵션 ON/OFF, 공유 트리, 메모리 회수 확인.
- 양색 메이트 전술, 규칙·반복·50수, perft 깊이3/4/5, 콘솔·자체 확인 스크립트 검사.
- 최종 ZIP을 다시 압축 해제한 후 전체 파일 해시와 모델 로딩 재확인.

[네이티브 실행](https://github.com/Blucy1004/Chromatic/actions/runs/40161264588) · [패키지 검사 JSON](reports/windows-package/40161264588-1/package.json) · [최종 ZIP 검사](reports/windows-package/40161264588-1/final-extraction-check.log)

## SHA-256

| 대상 | SHA-256 |
|---|---|
| ZIP | `33d42002cea835990e1207a064ca406901fad8479eb4d193ba8faf1f3953a5a8` |
| chromatic.exe | `cf3636f11dcea1c3be832c68b2c998171f4ca38a6433bb6b00bc927a6f243774` |
| default.zero / 실제 로딩 텐서 | `f6a00902c23e7f32f96b028ac36464cc9012cd8774d5e6b0dee5e1c94e1c03c9` |

소스 커밋: `caaff0fc9760baa0b05080cafe1286470470f93f`.
이 실행 파일은 정적 런타임·UTF-8 경로 처리·시작 입력 BOM 호환성을 포함한 패키지 빌드이므로, 이전 factorial 측정 실행 파일과 해시가 다르다. 이전 벤치마크를 이 바이너리의 재측정으로 표시하지 않는다.

이전 세 패키징 시도의 실패 기록은 `reports/windows-package/40160299948-1`, `40160543087-1`, `40160970360-1`에 보존했다. 자체 확인의 PowerShell 호환 문제를 수정했고, 통과하지 않은 패키지는 배포 ZIP으로 게시하지 않았다.
