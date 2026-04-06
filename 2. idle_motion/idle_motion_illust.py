"""
idle_motion_illust — 일러스트 타입 idle 애니메이션 생성 도구

일러스트(고해상도) 이미지용. 사인파 디스플레이스먼트로 계단 현상 없이
부드러운 팽창/수축 애니메이션을 생성한다.

도트(픽셀아트)는 idle_motion.py를 사용할 것.
"""

from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates
import os


def load_image(path: str) -> Image.Image:
    """이미지를 RGBA로 로드하고 불투명 영역만 크롭한다."""
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def apply_sine_displacement(img: Image.Image, top_ratio: float, bot_ratio: float,
                            sx: float) -> Image.Image:
    """사인파 디스플레이스먼트로 X축 스케일을 적용한다.

    리사이즈 대신 각 픽셀을 실수 좌표로 변위시켜 보간한다.
    영역 경계는 sin(π·t) 곡선으로 부드럽게 0→1→0 전환.
    계단 현상이 발생하지 않는다.
    """
    if abs(sx - 1.0) < 0.001:
        return img

    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    top = int(h * top_ratio)
    bot = int(h * bot_ratio)

    if bot <= top:
        return img

    center_x = w / 2.0
    y_coords, x_coords = np.mgrid[top:bot, 0:w].astype(np.float64)

    # 사인파 강도: 영역 상하단에서 0, 중앙에서 1
    t = (y_coords - top) / max(1, bot - top)
    strength = np.sin(t * np.pi)
    local_sx = 1.0 + (sx - 1.0) * strength

    # 역매핑: 출력 좌표 → 원본 좌표
    src_x = center_x + (x_coords - center_x) / local_sx

    for c in range(4):  # RGBA
        arr[top:bot, :, c] = map_coordinates(
            arr[:, :, c], [y_coords, src_x],
            order=3, mode='constant', cval=0
        )

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def create_frame(src_img: Image.Image, x_off: int, y_off: int,
                 scale_y: float, body_sx: float, body_range: tuple,
                 emphasis_scales: list, pad: int) -> Image.Image:
    """한 프레임을 생성한다."""
    w, h = src_img.size
    canvas_w = w + pad * 2
    canvas_h = h + pad * 2

    # Y축 스케일
    new_h = int(h * scale_y)
    scaled = src_img.resize((w, new_h), Image.LANCZOS)

    # 몸 전체 X축 스케일 (사인파 디스플레이스먼트)
    scaled = apply_sine_displacement(scaled, body_range[0], body_range[1], body_sx)

    # 강조 영역 X축 스케일 (사인파 디스플레이스먼트)
    for top_r, bot_r, sx in emphasis_scales:
        scaled = apply_sine_displacement(scaled, top_r, bot_r, sx)

    # 캔버스에 배치 (하단 고정)
    frame = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    paste_x = (canvas_w - w) // 2 + x_off
    paste_y = canvas_h - pad - new_h + y_off
    frame.paste(scaled, (paste_x, paste_y), scaled)
    return frame


def make_idle(input_path: str, output_path: str,
              num_frames: int = 4, duration: int = 300, pad: int = 15,
              y_offsets: list = None, x_offsets: list = None,
              scale_y_list: list = None,
              body_range: tuple = (0.10, 0.75), body_scale_x: list = None,
              emphasis_ranges: list = None) -> None:
    """idle 애니메이션을 생성한다."""
    src = load_image(input_path)

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
        print("사용법: python idle_motion_illust.py <입력 이미지> <출력 GIF> [설정 JSON]")
        print()
        print("일러스트(고해상도) 전용. 도트는 idle_motion.py를 사용할 것.")
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
