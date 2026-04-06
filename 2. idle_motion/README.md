# idle_motion — 스프라이트 idle 애니메이션 생성 도구

단일 이미지에서 숨쉬기/요동 idle 애니메이션을 생성한다.

## 원리

1. **전체 Y축 스케일** — 발 기준 하단 고정, 미세한 상하 요동
2. **전체 X축 팽창/수축** — 몸 전체가 미세하게 부풀었다 줄었다
3. **강조 영역 X축** — 가슴 등 특정 y 구간을 더 도드라지게 팽창/수축
4. **가우시안 분포** — 영역 중심이 가장 강하고 가장자리로 갈수록 약해짐

## 사용법

### 기본 (전체 요동만)
```bash
python idle_motion.py input.png output.gif
```

### 설정 JSON으로 커스텀
```bash
python idle_motion.py input.png output.gif config.json
```

### 설정 JSON 예시
```json
{
  "num_frames": 4,
  "duration": 300,
  "y_offsets": [0, -1, 0, 0],
  "scale_y": [1.0, 1.003, 1.0, 0.998],
  "body_range": [0.10, 0.75],
  "body_scale_x": [1.0, 1.008, 1.0, 0.992],
  "emphasis": [
    {"range": [0.22, 0.42], "scales": [1.0, 1.035, 1.0, 0.965]}
  ]
}
```

### 파라미터 설명

| 항목 | 설명 | 기본값 |
|---|---|---|
| `num_frames` | 프레임 수 | 4 |
| `duration` | 프레임당 ms | 300 |
| `y_offsets` | 프레임별 Y 이동 (px) | [0, -1, 0, 0] |
| `x_offsets` | 프레임별 X 이동 (px) | [0, 0, 0, 0] |
| `scale_y` | 프레임별 Y축 스케일 | [1.0, 1.003, 1.0, 0.998] |
| `body_range` | 몸 전체 X 스케일 범위 (top%, bot%) | [0.10, 0.75] |
| `body_scale_x` | 프레임별 몸 X축 스케일 | [1.0, 1.008, 1.0, 0.992] |
| `emphasis` | 강조 영역 리스트 | [] |

## 출력

- `output.gif` — 애니메이션 GIF
- `output_sheet.png` — 스프라이트시트 (가로 배치)

## GIF 저장 주의사항

GIF는 반투명(alpha 중간값)을 지원하지 않는다. 선따기된 이미지는 경계면에 반투명 픽셀이 있으므로, GIF 저장 시 반드시 **불투명 배경 위에 합성**한 뒤 RGB로 저장해야 한다.

```python
# 올바른 방식 (배경 합성)
for f in frames:
    bg = Image.new("RGB", f.size, (200, 200, 200))
    bg.paste(f, (0, 0), f)
    gif_frames.append(bg)
```

팔레트 변환이나 RGBA 직접 저장은 깨짐/깜빡임의 원인이 된다. 반투명을 유지하려면 GIF 대신 APNG 또는 스프라이트시트 PNG를 사용할 것.

## 요구 사항

- Python 3.10+
- Pillow, NumPy
