# clean_cut — 스프라이트 선따기 도구

흰색 배경 위의 2D 게임 스프라이트를 자동으로 오려내는 도구입니다.

## 동작 원리

1. **BFS 배경 탐색** — 마스크(올가미)로 지정한 영역에서 출발, 흰색 픽셀을 BFS로 탐색하여 배경 마킹
2. **배경 삭제** — 마킹된 배경 픽셀의 알파를 0으로 설정 (부드러운 제거 시 명도 비례 graduated alpha)
3. **경계면 알파 처리** — 오브젝트와 배경이 맞닿은 경계 픽셀에 명도 기반 투명도 적용

## 주요 기능

- **영역 제한 실행** — BFS가 마스크 영역 안에서만 동작 (머리카락 등 세밀 작업)
- **부드러운 제거** — 단계적 BFS (50~threshold, 5씩 증가) + 명도 비례 alpha
- **흰색 오차** — BFS 흰색 판별 허용 오차 조절 (0~255)
- **올가미 / 올가미 지우개** — 다각형 마스크 그리기/삭제
- **결과 → 작업대** — 반복 작업 지원 (줌/팬 유지)

## 파일 구성

| 파일 | 역할 |
|------|------|
| `clean_cut.py` | Python 알고리즘 코어 + HTTP 서버 + CLI |
| `index.html` | 웹 UI (Python 서버 필요) |
| `../clean_cut.html` | standalone 버전 (서버 불필요, JS 포팅) |
| `test_clean_cut.py` | pytest 테스트 |

## 사용법

```bash
# standalone — 브라우저에서 직접 열기
open ../clean_cut.html

# Python 서버
python clean_cut.py serve [port]

# CLI
python clean_cut.py <입력> <출력.png> [흰색오차] [마스크]
```

## 요구 사항

- standalone: 최신 브라우저
- Python: 3.10+, Pillow, NumPy
