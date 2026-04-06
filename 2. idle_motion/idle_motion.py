"""
idle_motion — 스프라이트 idle 애니메이션 생성 도구

단일 이미지에서 숨쉬기/요동 idle 애니메이션을 생성한다.
- 전체 Y축 스케일 (발 기준 하단 고정)
- 전체 X축 미세 팽창/수축
- 지정 영역 X축 강조 팽창/수축 (가슴 등)
"""

from PIL import Image
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


def apply_x_scale(arr: np.ndarray, top_ratio: float, bot_ratio: float,
                  sx: float, width: int) -> np.ndarray:
    """지정 y 영역에 가우시안 분포로 X축 스케일을 적용한다."""
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

        row = Image.fromarray(arr[y:y+1, :, :])
        new_w = int(width * local_sx)
        row_scaled = row.resize((new_w, 1), Image.LANCZOS)
        row_arr = np.array(row_scaled)

        offset = (new_w - width) // 2
        if new_w >= width:
            arr[y, :, :] = row_arr[0, offset:offset+width, :]
        else:
            start = (width - new_w) // 2
            arr[y, :, :] = 0
            arr[y, start:start+new_w, :] = row_arr[0, :, :]

    return arr


def create_frame(src_img: Image.Image, x_off: int, y_off: int,
                 scale_y: float, x_scales: list, pad: int) -> Image.Image:
    """한 프레임을 생성한다.

    Args:
        src_img: 원본 RGBA 이미지
        x_off: X축 정수 오프셋
        y_off: Y축 정수 오프셋
        scale_y: Y축 스케일 (1.0 = 원본)
        x_scales: [(top_ratio, bot_ratio, scale_x), ...] 영역별 X축 스케일
        pad: 캔버스 여백
    """
    w, h = src_img.size
    canvas_w = w + pad * 2
    canvas_h = h + pad * 2

    # Y축 스케일
    new_h = int(h * scale_y)
    scaled = src_img.resize((w, new_h), Image.LANCZOS)
    result_arr = np.array(scaled)

    # X축 스케일 (영역별, 순서대로 적용)
    for top_r, bot_r, sx in x_scales:
        result_arr = apply_x_scale(result_arr, top_r, bot_r, sx, w)

    result_img = Image.fromarray(result_arr, "RGBA")

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
        # X축 스케일 목록 구성
        x_scales = [(body_range[0], body_range[1], body_scale_x[i])]
        for emp in emphasis_ranges:
            x_scales.append((emp["range"][0], emp["range"][1], emp["scales"][i]))

        frame = create_frame(
            src, x_offsets[i], y_offsets[i],
            scale_y_list[i], x_scales, pad
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
