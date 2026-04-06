"""4가지 계단 현상 해결 방안 비교 테스트"""

from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
import json

PY_IMG = "ninja_nsfw_cut.png"
CONFIG = "config_ninja_nsfw.json"


def load_image(path):
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def apply_body_scale_x_resize(img, top_r, bot_r, sx):
    """기존 방식: 2D 블록 리사이즈 (계단 발생)"""
    if abs(sx - 1.0) < 0.001:
        return img
    w, h = img.size
    top, bot = int(h * top_r), int(h * bot_r)
    rh = bot - top
    if rh <= 0:
        return img
    region = img.crop((0, top, w, bot))
    new_w = max(1, int(w * sx))
    region_scaled = region.resize((new_w, rh), Image.LANCZOS)
    result = img.copy()
    clear = Image.new("RGBA", (w, rh), (0, 0, 0, 0))
    result.paste(clear, (0, top))
    ox = (w - new_w) // 2
    result.paste(region_scaled, (ox, top), region_scaled)
    return result


# === 방안 1: 디스플레이스먼트 맵 (map_coordinates 보간) ===
def apply_scale_method1(img, top_r, bot_r, sx):
    if abs(sx - 1.0) < 0.001:
        return img
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    top, bot = int(h * top_r), int(h * bot_r)
    center_x = w / 2.0

    y_coords, x_coords = np.mgrid[top:bot, 0:w].astype(np.float64)
    src_x = center_x + (x_coords - center_x) / sx

    for c in range(4):
        arr[top:bot, :, c] = map_coordinates(
            arr[:, :, c], [y_coords, src_x], order=3, mode='constant', cval=0
        )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


# === 방안 2: 슈퍼샘플링 (4x 업 → 스케일 → 다운) ===
def apply_scale_method2(img, top_r, bot_r, sx, factor=4):
    if abs(sx - 1.0) < 0.001:
        return img
    w, h = img.size
    big = img.resize((w * factor, h * factor), Image.LANCZOS)
    big = apply_body_scale_x_resize(big, top_r, bot_r, sx)
    return big.resize((w, h), Image.LANCZOS)


# === 방안 3: 사인파 디스플레이스먼트 (연속 곡선 변위) ===
def apply_scale_method3(img, top_r, bot_r, sx):
    if abs(sx - 1.0) < 0.001:
        return img
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    top, bot = int(h * top_r), int(h * bot_r)
    center_x = w / 2.0

    y_coords, x_coords = np.mgrid[top:bot, 0:w].astype(np.float64)

    # 사인파로 부드러운 0→1→0 강도
    t = (y_coords - top) / max(1, bot - top)
    strength = np.sin(t * np.pi)
    local_sx = 1.0 + (sx - 1.0) * strength

    src_x = center_x + (x_coords - center_x) / local_sx

    for c in range(4):
        arr[top:bot, :, c] = map_coordinates(
            arr[:, :, c], [y_coords, src_x], order=3, mode='constant', cval=0
        )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


# === 방안 4: 외곽선 분리 후 재합성 ===
def apply_scale_method4(img, top_r, bot_r, sx):
    if abs(sx - 1.0) < 0.001:
        return img
    arr = np.array(img)
    h, w = arr.shape[:2]
    top, bot = int(h * top_r), int(h * bot_r)

    # 외곽선 검출 (알파 채널 기반 + 내부 선)
    gray = cv2.cvtColor(arr[top:bot], cv2.COLOR_RGBA2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # 외곽선 약간 두껍게
    kernel = np.ones((2, 2), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)

    # 외곽선 픽셀 저장
    edge_pixels = arr[top:bot].copy()
    edge_pixels[edge_mask == 0] = [0, 0, 0, 0]

    # 내부를 디스플레이스먼트로 스케일 (방안1과 동일)
    float_arr = arr.astype(np.float64)
    center_x = w / 2.0
    y_coords, x_coords = np.mgrid[top:bot, 0:w].astype(np.float64)
    src_x = center_x + (x_coords - center_x) / sx

    for c in range(4):
        float_arr[top:bot, :, c] = map_coordinates(
            float_arr[:, :, c], [y_coords, src_x], order=3, mode='constant', cval=0
        )

    result = np.clip(float_arr, 0, 255).astype(np.uint8)

    # 외곽선도 같은 변위로 이동
    edge_float = edge_pixels.astype(np.float64)
    edge_warped = np.zeros_like(edge_float)
    for c in range(4):
        edge_warped[:, :, c] = map_coordinates(
            edge_float[:, :, c],
            [y_coords - top, src_x],
            order=1, mode='constant', cval=0
        )

    # 외곽선 합성 (외곽선 알파가 있는 곳은 외곽선 우선)
    edge_warped = np.clip(edge_warped, 0, 255).astype(np.uint8)
    e_alpha = edge_warped[:, :, 3:4].astype(np.float64) / 255.0
    blended = result[top:bot].astype(np.float64) * (1 - e_alpha) + edge_warped.astype(np.float64) * e_alpha
    result[top:bot] = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(result, "RGBA")


def create_frame(src_img, x_off, y_off, scale_y, body_sx, body_range,
                 emphasis_scales, pad, scale_fn):
    w, h = src_img.size
    canvas_w = w + pad * 2
    canvas_h = h + pad * 2

    new_h = int(h * scale_y)
    scaled = src_img.resize((w, new_h), Image.LANCZOS)

    # body
    scaled = scale_fn(scaled, body_range[0], body_range[1], body_sx)

    # emphasis
    for top_r, bot_r, sx in emphasis_scales:
        scaled = scale_fn(scaled, top_r, bot_r, sx)

    frame = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    paste_x = (canvas_w - w) // 2 + x_off
    paste_y = canvas_h - pad - new_h + y_off
    scaled_img = scaled if isinstance(scaled, Image.Image) else Image.fromarray(scaled, "RGBA")
    frame.paste(scaled_img, (paste_x, paste_y), scaled_img)
    return frame


def generate_gif(src, config, output_path, scale_fn):
    nf = config.get("num_frames", 4)
    dur = config.get("duration", 300)
    pad = 15
    y_off = config.get("y_offsets", [0, -1, 0, 0])
    x_off = config.get("x_offsets", [0] * nf)
    sy = config.get("scale_y", [1.0, 1.003, 1.0, 0.998])
    br = tuple(config.get("body_range", [0.10, 0.75]))
    bsx = config.get("body_scale_x", [1.0, 1.008, 1.0, 0.992])
    emps = config.get("emphasis", [])

    frames = []
    for i in range(nf):
        emp_scales = [(e["range"][0], e["range"][1], e["scales"][i]) for e in emps]
        f = create_frame(src, x_off[i], y_off[i], sy[i], bsx[i], br, emp_scales, pad, scale_fn)
        frames.append(f)

    gif_frames = []
    for f in frames:
        bg = Image.new("RGB", f.size, (200, 200, 200))
        bg.paste(f, (0, 0), f)
        gif_frames.append(bg)

    gif_frames[0].save(output_path, save_all=True, append_images=gif_frames[1:],
                       duration=dur, loop=0)

    # 축소본
    small_frames = []
    for gf in gif_frames:
        w, h = gf.size
        small_frames.append(gf.resize((w // 2, h // 2), Image.LANCZOS))
    small_path = output_path.replace(".gif", "_small.gif")
    small_frames[0].save(small_path, save_all=True, append_images=small_frames[1:],
                         duration=dur, loop=0)
    print(f"완료: {output_path}")


if __name__ == "__main__":
    src = load_image(PY_IMG)
    with open(CONFIG) as f:
        config = json.load(f)

    methods = [
        ("method1_displacement.gif", apply_scale_method1),
        ("method2_supersample.gif", apply_scale_method2),
        ("method3_sine_disp.gif", apply_scale_method3),
        ("method4_outline.gif", apply_scale_method4),
    ]

    for name, fn in methods:
        print(f"생성 중: {name}")
        generate_gif(src, config, name, fn)
