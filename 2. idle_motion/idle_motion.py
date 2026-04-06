"""
idle_motion — 스프라이트 idle 애니메이션 생성 도구

단일 이미지에서 숨쉬기/요동 idle 애니메이션을 생성한다.
- 전체 Y축 스케일 (발 기준 하단 고정)
- 전체 X축 미세 팽창/수축
- 지정 영역 X축 강조 팽창/수축 (가슴 등)
"""

from PIL import Image, ImageFilter
import numpy as np
import math
import os


def load_image(path: str) -> Image.Image:
    """이미지를 RGBA로 로드하고 불투명 영역만 크롭한다."""
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def _find_largest_span(alpha_row: np.ndarray, threshold: int = 10):
    """행에서 가장 넓은 불투명 연속 구간(몸통)의 (start, end)를 반환한다."""
    opaque = alpha_row >= threshold
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    in_span = False

    for x in range(len(opaque)):
        if opaque[x]:
            if not in_span:
                cur_start = x
                cur_len = 0
                in_span = True
            cur_len += 1
        else:
            if in_span:
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                in_span = False
    if in_span and cur_len > best_len:
        best_start, best_len = cur_start, cur_len

    if best_len == 0:
        return None
    return best_start, best_start + best_len


def apply_x_scale(arr: np.ndarray, top_ratio: float, bot_ratio: float,
                  sx: float, width: int) -> np.ndarray:
    """지정 y 영역에 가우시안 분포로 X축 스케일을 적용한다.
    각 행에서 가장 넓은 불투명 덩어리(몸통)만 스케일하고 나머지(팔 등)는 유지."""
    h = arr.shape[0]
    top = int(h * top_ratio)
    bot = int(h * bot_ratio)
    mid = (top + bot) / 2.0
    half = (bot - top) / 2.0

    if half == 0 or sx == 1.0:
        return arr

    for y in range(top, bot):
        dist = abs(y - mid) / half
        strength = max(0.0, 1.0 - dist * dist)
        local_sx = 1.0 + (sx - 1.0) * strength

        if abs(local_sx - 1.0) < 0.001:
            continue

        # 가장 넓은 불투명 구간(몸통) 찾기
        span = _find_largest_span(arr[y, :, 3])
        if span is None:
            continue
        s_start, s_end = span
        s_width = s_end - s_start

        # 몸통 부분만 추출해서 스케일
        body = Image.fromarray(arr[y:y+1, s_start:s_end, :])
        new_bw = max(1, int(s_width * local_sx))
        body_scaled = np.array(body.resize((new_bw, 1), Image.LANCZOS))[0]

        # 몸통 영역을 새 크기로 교체 (중심 정렬)
        body_center = (s_start + s_end) // 2
        new_start = body_center - new_bw // 2
        new_end = new_start + new_bw

        # 원본 행 백업 (팔 등 보존용)
        row_backup = arr[y, :, :].copy()
        # 몸통 원래 자리 지우기
        arr[y, s_start:s_end, :] = 0

        # 클리핑
        src_l = max(0, -new_start)
        src_r = new_bw - max(0, new_end - width)
        dst_l = max(0, new_start)
        dst_r = min(width, new_end)

        if dst_r > dst_l and src_r > src_l:
            arr[y, dst_l:dst_r, :] = body_scaled[src_l:src_r, :]

        # 팔 등 몸통 바깥 영역 복원
        arr[y, :s_start, :] = row_backup[:s_start, :]
        if s_end < width:
            arr[y, s_end:, :] = row_backup[s_end:, :]

    return arr


def apply_body_scale_x(img: Image.Image, top_ratio: float, bot_ratio: float,
                       sx: float) -> Image.Image:
    """몸 전체 X축 스케일을 2D 영역 통째로 적용한다 (울렁임 방지)."""
    if abs(sx - 1.0) < 0.001:
        return img

    w, h = img.size
    top = int(h * top_ratio)
    bot = int(h * bot_ratio)
    region_h = bot - top
    if region_h <= 0:
        return img

    # 영역 추출 → 통째 리사이즈 → 중심 정렬로 붙여넣기
    region = img.crop((0, top, w, bot))
    new_w = max(1, int(w * sx))
    region_scaled = region.resize((new_w, region_h), Image.LANCZOS)

    result = img.copy()
    # 영역 지우기
    clear = Image.new("RGBA", (w, region_h), (0, 0, 0, 0))
    result.paste(clear, (0, top))
    # 중심 정렬로 붙이기
    offset_x = (w - new_w) // 2
    result.paste(region_scaled, (offset_x, top), region_scaled)

    return result


def create_frame(src_img: Image.Image, x_off: int, y_off: int,
                 scale_y: float, body_sx: float, body_range: tuple,
                 emphasis_scales: list, pad: int) -> Image.Image:
    """한 프레임을 생성한다.

    Args:
        src_img: 원본 RGBA 이미지
        x_off: X축 정수 오프셋
        y_off: Y축 정수 오프셋
        scale_y: Y축 스케일 (1.0 = 원본)
        body_sx: 몸 전체 X축 스케일 (2D 통째 리사이즈)
        body_range: 몸 전체 X축 스케일 영역 (top_ratio, bot_ratio)
        emphasis_scales: [(top_ratio, bot_ratio, scale_x), ...] 강조 영역 X축 스케일
        pad: 캔버스 여백
    """
    w, h = src_img.size
    canvas_w = w + pad * 2
    canvas_h = h + pad * 2

    # Y축 스케일
    new_h = int(h * scale_y)
    scaled = src_img.resize((w, new_h), Image.LANCZOS)

    # 몸 전체 X축 스케일 (2D 통째)
    scaled = apply_body_scale_x(scaled, body_range[0], body_range[1], body_sx)

    # 강조 영역 X축 스케일 (2D 통째)
    for top_r, bot_r, sx in emphasis_scales:
        scaled = apply_body_scale_x(scaled, top_r, bot_r, sx)

    # 불투명 영역에만 가우시안 블러 (계단 현상 완화)
    arr = np.array(scaled)
    blurred = scaled.filter(ImageFilter.GaussianBlur(radius=0.8))
    blur_arr = np.array(blurred)
    # 알파 > 0 인 픽셀(PNG 객체)만 블러 적용, 투명 영역은 그대로
    mask = arr[:, :, 3] > 0
    arr[mask] = blur_arr[mask]
    result_img = Image.fromarray(arr, "RGBA")

    # 캔버스에 배치 (하단 고정)
    frame = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    paste_x = (canvas_w - w) // 2 + x_off
    paste_y = canvas_h - pad - new_h + y_off
    frame.paste(result_img, (paste_x, paste_y), result_img)
    return frame


def make_idle(input_path: str, output_path: str,
              num_frames: int = 4, duration: int = 300, pad: int = 15,
              y_offsets: list = None, x_offsets: list = None,
              scale_y_list: list = None,
              body_range: tuple = (0.10, 0.75), body_scale_x: list = None,
              emphasis_ranges: list = None) -> None:
    """idle 애니메이션을 생성한다.

    Args:
        input_path: 입력 이미지 경로 (RGBA PNG 권장)
        output_path: 출력 GIF 경로
        num_frames: 프레임 수 (기본 4)
        duration: 프레임당 ms (기본 300)
        pad: 캔버스 여백 px
        y_offsets: 프레임별 Y 오프셋 리스트
        x_offsets: 프레임별 X 오프셋 리스트
        scale_y_list: 프레임별 Y축 스케일 리스트
        body_range: 몸 전체 X축 스케일 영역 (top_ratio, bot_ratio)
        body_scale_x: 프레임별 몸 전체 X축 스케일 리스트
        emphasis_ranges: 강조 영역 리스트
            [{'range': (top_ratio, bot_ratio), 'scales': [프레임별 스케일]}, ...]
    """
    src = load_image(input_path)

    # 기본값
    if y_offsets is None:
        y_offsets = [0, -1, 0, 0]
    if x_offsets is None:
        x_offsets = [0] * num_frames
    if scale_y_list is None:
        scale_y_list = [1.0, 1.003, 1.0, 0.998]
    if body_scale_x is None:
        body_scale_x = [1.0, 1.008, 1.0, 0.992]
    if emphasis_ranges is None:
        emphasis_ranges = []

    frames = []
    for i in range(num_frames):
        # 강조 영역 X축 스케일 목록
        emp_scales = []
        for emp in emphasis_ranges:
            emp_scales.append((emp["range"][0], emp["range"][1], emp["scales"][i]))

        frame = create_frame(
            src, x_offsets[i], y_offsets[i],
            scale_y_list[i], body_scale_x[i], body_range,
            emp_scales, pad
        )
        frames.append(frame)

    # GIF 저장 (배경 합성 — GIF는 반투명 미지원)
    gif_frames = []
    for f in frames:
        bg = Image.new("RGB", f.size, (200, 200, 200))
        bg.paste(f, (0, 0), f)
        gif_frames.append(bg)

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=0,
    )

    # 스프라이트시트도 저장
    sheet_path = output_path.rsplit(".", 1)[0] + "_sheet.png"
    cw, ch = frames[0].size
    sheet = Image.new("RGBA", (cw * num_frames, ch), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, (cw * i, 0), f)
    sheet.save(sheet_path)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("사용법: python idle_motion.py <입력 이미지> <출력 GIF> [설정 JSON]")
        print()
        print("설정 JSON 예시:")
        print(json.dumps({
            "num_frames": 4,
            "duration": 300,
            "y_offsets": [0, -1, 0, 0],
            "scale_y": [1.0, 1.003, 1.0, 0.998],
            "body_range": [0.10, 0.75],
            "body_scale_x": [1.0, 1.008, 1.0, 0.992],
            "emphasis": [
                {"range": [0.22, 0.42], "scales": [1.0, 1.035, 1.0, 0.965]}
            ]
        }, indent=2, ensure_ascii=False))
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    # 기본 설정
    kwargs = {}

    if len(sys.argv) > 3:
        with open(sys.argv[3], "r") as f:
            config = json.load(f)

        if "num_frames" in config:
            kwargs["num_frames"] = config["num_frames"]
        if "duration" in config:
            kwargs["duration"] = config["duration"]
        if "y_offsets" in config:
            kwargs["y_offsets"] = config["y_offsets"]
        if "x_offsets" in config:
            kwargs["x_offsets"] = config["x_offsets"]
        if "scale_y" in config:
            kwargs["scale_y_list"] = config["scale_y"]
        if "body_range" in config:
            kwargs["body_range"] = tuple(config["body_range"])
        if "body_scale_x" in config:
            kwargs["body_scale_x"] = config["body_scale_x"]
        if "emphasis" in config:
            kwargs["emphasis_ranges"] = [
                {"range": tuple(e["range"]), "scales": e["scales"]}
                for e in config["emphasis"]
            ]

    make_idle(in_path, out_path, **kwargs)
    print(f"완료: {out_path}")
