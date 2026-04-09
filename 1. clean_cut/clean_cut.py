"""
clean_cut — 스프라이트 선따기 도구

마스크 ROI에서 시작하는 BFS flood-fill로 흰 배경을 제거하고,
경계면 픽셀에 명도+채도 기반 알파를 적용해 깔끔하게 오려낸다.

테두리 자동 BFS는 사용하지 않는다 — 사용자가 마스크로 명시적으로
지정한 영역만 처리한다.

CLI:
  python clean_cut.py serve [port]                       웹 인터페이스 시작
  python clean_cut.py <input> <output> [thresh] [mask]   파일 모드
"""

import base64
import json
import os
import sys
from collections import deque
from io import BytesIO

import numpy as np
from PIL import Image


# ============================================================
# 픽셀 판별
# ============================================================

def is_white(pixel_rgb, threshold: int = 10) -> bool:
    """RGB 값이 흰색에 충분히 가까운지 판별한다."""
    r, g, b = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])
    return (255 - r) <= threshold and (255 - g) <= threshold and (255 - b) <= threshold


# ============================================================
# I/O
# ============================================================

def load_rgba(path: str) -> np.ndarray:
    """이미지를 RGBA numpy 배열로 로드한다."""
    img = Image.open(path).convert("RGBA")
    return np.array(img)


def save_png(pixels: np.ndarray, path: str) -> None:
    """RGBA numpy 배열을 PNG로 저장한다."""
    Image.fromarray(pixels, "RGBA").save(path)


def save_psd(pixels: np.ndarray, path: str) -> None:
    """RGBA 결과를 PSD로 저장한다. 확인용 보색 배경 레이어 포함."""
    from psd_tools import PSDImage  # 옵션 의존성

    h, w = pixels.shape[:2]
    psd = PSDImage.new("RGBA", (w, h))

    opaque = pixels[:, :, 3] > 0
    if np.any(opaque):
        avg_r = np.mean(pixels[opaque, 0])
        avg_g = np.mean(pixels[opaque, 1])
        avg_b = np.mean(pixels[opaque, 2])
    else:
        avg_r, avg_g, avg_b = 128, 128, 128
    comp_r, comp_g, comp_b = 255 - int(avg_r), 255 - int(avg_g), 255 - int(avg_b)

    bg = np.full((h, w, 4), 255, dtype=np.uint8)
    bg[:, :, 0] = comp_r
    bg[:, :, 1] = comp_g
    bg[:, :, 2] = comp_b

    cut_layer = psd.create_pixel_layer(name="cut", image=Image.fromarray(pixels, "RGBA"))
    bg_layer = psd.create_pixel_layer(name="bg", image=Image.fromarray(bg, "RGBA"))
    psd.append(bg_layer)
    psd.append(cut_layer)
    psd.save(path)


# ============================================================
# Flood-fill 코어
# ============================================================

def _flood_from_seed(
    pixels: np.ndarray,
    visited: np.ndarray,
    sy: int,
    sx: int,
    threshold: int = 10,
) -> None:
    """단일 시작점에서 흰색 인접 영역을 BFS flood-fill하여 visited를 in-place로 갱신.

    시작점이 이미 visited이거나 흰색이 아니면 아무 일도 하지 않는다.
    """
    if visited[sy, sx]:
        return
    if not is_white(pixels[sy, sx, :3], threshold):
        return

    h, w = pixels.shape[:2]
    visited[sy, sx] = True
    queue: deque = deque([(sy, sx)])
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if is_white(pixels[ny, nx, :3], threshold):
                    visited[ny, nx] = True
                    queue.append((ny, nx))


def mark_background_from_mask(
    pixels: np.ndarray,
    bg_mask: np.ndarray,
    mask_pixels: np.ndarray,
    threshold: int = 10,
    mask_threshold: int = 10,
) -> np.ndarray:
    """마스크 이미지의 ROI 픽셀을 시작점으로 BFS flood-fill을 실행하여 bg_mask를 갱신한다.

    visited 배열을 공유하므로 한 번 채운 영역은 다시 탐색하지 않는다.
    이 함수가 유일한 배경 마킹 경로 — 테두리 자동 BFS는 더 이상 사용하지 않는다.

    마스크 ROI 판정 (자동 인식):
    - HTML 그림판 마스크: 알파>0인 픽셀 (그려진 부분)
    - 옛 검정 마스크: RGB가 충분히 어두운 픽셀
    """
    h, w = pixels.shape[:2]
    if mask_pixels.shape[:2] != (h, w):
        raise ValueError(
            f"마스크 크기 {mask_pixels.shape[:2]}가 원본 {(h, w)}와 다름"
        )

    mask_alpha = mask_pixels[..., 3]
    if (mask_alpha < 255).any():
        is_roi = mask_alpha > mask_threshold
    else:
        mask_rgb = mask_pixels[..., :3]
        is_roi = mask_rgb.max(axis=2) <= mask_threshold

    ys, xs = np.where(is_roi)
    for y, x in zip(ys, xs):
        _flood_from_seed(pixels, bg_mask, int(y), int(x), threshold)

    return bg_mask


# ============================================================
# 배경 제거 + 경계면 알파
# ============================================================

def remove_background(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """배경으로 마킹된 픽셀의 알파를 0으로 설정한다."""
    result = pixels.copy()
    result[mask, 3] = 0
    return result


def find_boundary(bg_mask: np.ndarray) -> np.ndarray:
    """오브젝트(비배경) 픽셀 중 배경과 인접한 픽셀을 경계면으로 식별한다."""
    h, w = bg_mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    obj_mask = ~bg_mask

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.zeros_like(bg_mask)
        sy = slice(max(0, dy), min(h, h + dy) or None)
        sx = slice(max(0, dx), min(w, w + dx) or None)
        oy = slice(max(0, -dy), min(h, h - dy) or None)
        ox = slice(max(0, -dx), min(w, w - dx) or None)
        shifted[oy, ox] = bg_mask[sy, sx]
        boundary |= (obj_mask & shifted)

    return boundary


def expand_boundary(boundary: np.ndarray, bg_mask: np.ndarray, depth: int = 1) -> np.ndarray:
    """경계면에서 오브젝트 안쪽으로 depth 픽셀만큼 확장한다."""
    expanded = boundary.copy()
    obj_mask = ~bg_mask

    for _ in range(depth):
        new_layer = np.zeros_like(expanded)
        h, w = expanded.shape
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.zeros_like(expanded)
            sy = slice(max(0, dy), min(h, h + dy) or None)
            sx = slice(max(0, dx), min(w, w + dx) or None)
            oy = slice(max(0, -dy), min(h, h - dy) or None)
            ox = slice(max(0, -dx), min(w, w - dx) or None)
            shifted[oy, ox] = expanded[sy, sx]
            new_layer |= shifted
        expanded |= (new_layer & obj_mask)

    return expanded


def apply_boundary_alpha(
    pixels: np.ndarray,
    boundary_mask: np.ndarray,
    strength: int = 60,
) -> np.ndarray:
    """경계면 픽셀에 명도 기반 알파를 적용한다.

    흰색에 가까울수록 투명. gamma 커브로 강도를 조절한다.
    strength가 높을수록 밝은 경계 픽셀도 공격적으로 투명화하고,
    기존 투명도도 더 깊어진다.

    gamma = strength / 30  (strength=30이면 선형, 60이면 제곱)
    alpha = (1 - brightness) ^ gamma * 255
    """
    result = pixels.copy()
    ys, xs = np.where(boundary_mask)
    if len(ys) == 0:
        return result

    r = result[ys, xs, 0].astype(float)
    g = result[ys, xs, 1].astype(float)
    b = result[ys, xs, 2].astype(float)

    brightness = (r + g + b) / (3.0 * 255.0)
    gamma = max(strength, 1) / 30.0
    alpha = (np.power(1.0 - brightness, gamma) * 255.0).clip(0, 255).astype(np.uint8)
    result[ys, xs, 3] = alpha
    return result


# ============================================================
# 파이프라인 — 단일 출처 (파일/HTTP 모두 호출)
# ============================================================

def run_pipeline(
    pixels: np.ndarray,
    mask_pixels=None,
    threshold: int = 10,
    boundary_depth: int = 1,
    boundary_strength: int = 60,
) -> np.ndarray:
    """numpy 배열만 다루는 알고리즘 핵심.

    파일 모드(clean_cut)와 HTTP 모드(_process_bytes)가 둘 다 이 함수를
    호출한다. 알고리즘 변경은 여기 한 곳만 손대면 모든 진입점에 반영된다.

    mask_pixels=None이면 bg_mask는 전부 False로 시작 → 어떤 배경 제거도
    일어나지 않음 (no-op). 반드시 마스크가 필요한 설계.
    """
    bg_mask = np.zeros(pixels.shape[:2], dtype=bool)
    if mask_pixels is not None:
        bg_mask = mark_background_from_mask(
            pixels, bg_mask, mask_pixels, threshold=threshold
        )
    result = remove_background(pixels, bg_mask)
    boundary = find_boundary(bg_mask)
    expanded = expand_boundary(boundary, bg_mask, depth=boundary_depth)
    result = apply_boundary_alpha(result, expanded, strength=boundary_strength)
    return result


def clean_cut(
    input_path: str,
    output_path: str,
    threshold: int = 10,
    boundary_depth: int = 1,
    mask_path=None,
) -> None:
    """파일 기반 진입점."""
    pixels = load_rgba(input_path)
    mask_pixels = load_rgba(mask_path) if mask_path else None
    result = run_pipeline(
        pixels,
        mask_pixels=mask_pixels,
        threshold=threshold,
        boundary_depth=boundary_depth,
    )
    if output_path.lower().endswith(".psd"):
        save_psd(result, output_path)
    else:
        save_png(result, output_path)


# ============================================================
# HTTP 서버 — 브라우저 그림판 인터페이스
# ============================================================

def _process_bytes(
    image_bytes: bytes,
    mask_bytes,
    boundary_strength: int = 60,
) -> bytes:
    """HTTP 요청 처리 — 바이트 IO 후 run_pipeline 호출."""
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    px = np.array(img)

    mask_px = None
    if mask_bytes:
        mask_img = Image.open(BytesIO(mask_bytes)).convert("RGBA")
        if mask_img.size != img.size:
            mask_img = mask_img.resize(img.size)
        mask_px = np.array(mask_img)

    result = run_pipeline(px, mask_pixels=mask_px, boundary_strength=boundary_strength)

    out = BytesIO()
    Image.fromarray(result).save(out, format="PNG")
    return out.getvalue()


def serve(port: int = 8765) -> None:
    """로컬 HTTP 서버 시작 — http://localhost:<port> 에서 그림판 UI 제공."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    def _strip_data_url(s: str) -> str:
        if not s:
            return ""
        return s.split(",", 1)[1] if "," in s else s

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                try:
                    with open(index_path, "rb") as f:
                        content = f.read()
                except FileNotFoundError:
                    self.send_error(404, "index.html not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/process":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                image_b64 = _strip_data_url(data.get("image", ""))
                mask_b64 = _strip_data_url(data.get("mask", ""))
                if not image_b64:
                    raise ValueError("image 필드 없음")
                image_bytes = base64.b64decode(image_b64)
                mask_bytes = base64.b64decode(mask_b64) if mask_b64 else None
                strength = int(data.get("boundary_strength", 60))

                result_bytes = _process_bytes(image_bytes, mask_bytes, boundary_strength=strength)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(err_msg.encode("utf-8"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(result_bytes)))
            self.end_headers()
            self.wfile.write(result_bytes)

        def log_message(self, format, *args):
            sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    server = HTTPServer(("localhost", port), Handler)
    print(f"clean_cut web — http://localhost:{port}")
    print("Ctrl+C로 종료")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
        server.server_close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] == "serve":
        port = int(args[1]) if len(args) > 1 else 8765
        serve(port=port)
    elif len(args) >= 2:
        in_path = args[0]
        out_path = args[1]
        thresh = int(args[2]) if len(args) > 2 else 10
        mask = args[3] if len(args) > 3 else None
        clean_cut(in_path, out_path, thresh, mask_path=mask)
        print(f"완료: {out_path}")
    else:
        print("사용법:")
        print("  python clean_cut.py serve [port]                       웹 서버 시작 (기본 8765)")
        print("  python clean_cut.py <in> <out> [thresh] [mask]         파일 모드")
        sys.exit(1)
