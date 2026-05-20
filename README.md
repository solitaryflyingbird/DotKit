# DotKit

A collection of tools for pixel art and dot graphics workflow.
*도트 그래픽 작업에 필요한 도구 모음입니다.*

---

## Tools

### 1. clean_cut — 스프라이트 선따기

흰색 배경 위의 2D 스프라이트를 자동으로 오려내는 도구.

#### 핵심 기능

- **BFS 배경 제거** — 마스크 영역에서 출발, 흰색 픽셀을 따라 flood-fill로 배경 탐색 및 제거
- **경계면 알파 처리** — 오브젝트와 배경이 맞닿은 경계 픽셀에 명도 기반 투명도 적용 (안티앨리어싱 자연 처리)
- **영역 제한 실행 (confined)** — BFS가 마스크 영역 밖으로 확장하지 않는 모드. 머리카락 등 세밀한 부분 다듬기용
- **부드러운 제거 (soft)** — 배경 영역에 하드컷(alpha=0) 대신 명도 비례 graduated alpha 적용
- **단계적 BFS** — 부드러운 제거 시 threshold를 50부터 5씩 올리며 반복. 어두운 픽셀이 벽 역할하여 오브젝트 침투 방지
- **흰색 오차 조절** — BFS 흰색 판별 허용 오차 (0~255). 높일수록 회색까지 배경으로 인식
- **경계 확장** — 배경 제거된 영역의 경계에서 안쪽으로 N픽셀 추가로 명도 기반 투명도 적용 (기본 1px). 안티앨리어싱 잔여물 정리용

#### 도구

- **올가미** — 다각형 영역을 그려 마스크 지정
- **올가미 지우개** — 마스크 영역 삭제
- **줌/팬** — 휠 줌, 스페이스+드래그 팬, 하단 줌 슬라이더
- **배경색 선택** — 체커보드 또는 커스텀 단색
- **결과 → 작업대** — 결과를 입력으로 되돌려 반복 작업 (줌/팬 유지)

#### 사용법

**standalone (서버 불필요):**
```
clean_cut.html 파일을 브라우저에서 직접 열기
```

**Python 서버 모드:**
```bash
python clean_cut.py serve [port]    # 기본 8765
```

**CLI 파일 모드:**
```bash
python clean_cut.py <입력> <출력.png> [흰색오차] [마스크]
```

#### 요구 사항

- standalone: 최신 브라우저 (Chrome, Safari, Firefox)
- Python 서버: Python 3.10+, Pillow, NumPy

---

### 2. idle_motion

> 준비 중. *Coming soon.*
