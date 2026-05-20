"""
clean_cut_one_touch — multiply-to-alpha 원터치 변환기

흰 배경 위의 이미지를 픽셀별로 처리해 alpha 합성이 multiply 블렌드와
동일한 결과를 내도록 RGBA로 변환한다. 마스크/BFS 없음 — 이미지 한 장만
들어가면 끝.

폴더 1(clean_cut)이 "사용자가 칠한 영역만 정밀 제거"라면,
이쪽은 "흰 배경 전제하에 전 픽셀 일괄 역산".

CLI:
  python clean_cut_one_touch.py serve [port]                       웹 인터페이스 시작
  python clean_cut_one_touch.py <input> <output> [mode] [thresh]   파일 모드
    mode    : auto | grayscale | tint | color  (기본 auto)
    thresh  : alpha 노이즈 컷오프 0~1 (기본 0.004)
"""

import base64
import json
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image


# ============================================================
# I/O
# ============================================================

def load_rgb(path: str) -> np.ndarray:
    """이미지를 RGB numpy 배열(uint8)로 로드. RGBA 입력은 흰 배경으로 flatten."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    return np.array(img)


def save_png(rgba: np.ndarray, path: str) -> None:
    """RGBA numpy 배열을 PNG로 저장."""
    out = rgba.copy()
    out[out[:, :, 3] == 0, :3] = 0
    Image.fromarray(out, "RGBA").save(path)


# ============================================================
# 자동 모드 판별
# ============================================================

def auto_detect_mode(rgb: np.ndarray, sat_threshold: float = 0.05, ratio: float = 0.95) -> str:
    """채도 히스토그램으로 grayscale vs color 자동 판별.

    불투명(=비흰색) 픽셀 중 ratio 비율 이상이 채도 < sat_threshold이면
    grayscale로 간주.
    """
    f = rgb.astype(np.float32) / 255.0
    max_ch = f.max(axis=2)
    min_ch = f.min(axis=2)
    saturation = np.where(max_ch > 1e-6, (max_ch - min_ch) / np.maximum(max_ch, 1e-6), 0.0)

    # 거의 흰색 픽셀은 제외
    non_white = max_ch < 0.98
    if not non_white.any():
        return "grayscale"

    low_sat = (saturation < sat_threshold) & non_white
    frac = low_sat.sum() / non_white.sum()
    return "grayscale" if frac >= ratio else "color"


# ============================================================
# 배경색 정규화
# ============================================================

def normalize_bg(rgb: np.ndarray, bg_color: tuple) -> np.ndarray:
    """bg_color 배경을 #FFFFFF로 정규화한 float32 배열을 반환.

    (R - Br) / (255 - Br) 식으로 픽셀을 흰 배경 기준으로 재투영.
    bg_color == (255,255,255)이면 단순 정규화.
    """
    f = rgb.astype(np.float32)
    br, bg_g, bb = float(bg_color[0]), float(bg_color[1]), float(bg_color[2])

    def channel(c, b):
        denom = 255.0 - b
        if denom < 1.0:
            return np.clip(c / 255.0, 0.0, 1.0)
        return np.clip((c - b) / denom, 0.0, 1.0)

    r = channel(f[:, :, 0], br)
    g = channel(f[:, :, 1], bg_g)
    b = channel(f[:, :, 2], bb)
    return np.stack([r, g, b], axis=-1)


# ============================================================
# 핵심 변환 — Mode A/B/C
# ============================================================

def multiply_to_alpha(
    rgb: np.ndarray,
    mode: str = "auto",
    tint_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
    threshold: float = 0.004,
) -> np.ndarray:
    """흰 배경 RGB → straight alpha RGBA 변환.

    mode:
      grayscale — 단일 색(tint_color) + 밝기 기반 alpha. 흑백 원본에 정확.
      tint      — grayscale과 동일하나 의도가 명시적. tint_color 필수.
      color     — 픽셀별 RGB+alpha 역산. 풀컬러 근사.
      auto      — 채도로 grayscale/color 자동 선택.

    threshold: alpha < threshold 인 픽셀은 완전 투명으로 클램프 (노이즈 제거).
    """
    f = normalize_bg(rgb, bg_color)

    actual_mode = mode
    if actual_mode == "auto":
        actual_mode = auto_detect_mode(rgb)

    if actual_mode in ("grayscale", "tint"):
        # ITU-R BT.601 휘도
        lum = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        alpha = 1.0 - lum
        r = np.full_like(alpha, tint_color[0] / 255.0)
        g = np.full_like(alpha, tint_color[1] / 255.0)
        b = np.full_like(alpha, tint_color[2] / 255.0)
    elif actual_mode == "color":
        min_ch = np.minimum(np.minimum(f[:, :, 0], f[:, :, 1]), f[:, :, 2])
        alpha = 1.0 - min_ch
        safe = np.where(alpha > threshold, alpha, 1.0)
        r = 1.0 - (1.0 - f[:, :, 0]) / safe
        g = 1.0 - (1.0 - f[:, :, 1]) / safe
        b = 1.0 - (1.0 - f[:, :, 2]) / safe
    else:
        raise ValueError(f"알 수 없는 모드: {mode}")

    # 노이즈 컷오프
    transparent = alpha < threshold
    alpha = np.where(transparent, 0.0, alpha)
    r = np.where(transparent, 0.0, r)
    g = np.where(transparent, 0.0, g)
    b = np.where(transparent, 0.0, b)

    out = np.stack([
        np.clip(r, 0.0, 1.0),
        np.clip(g, 0.0, 1.0),
        np.clip(b, 0.0, 1.0),
        np.clip(alpha, 0.0, 1.0),
    ], axis=-1)
    return (out * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# 파이프라인 — 단일 출처
# ============================================================

def run_pipeline(
    rgb: np.ndarray,
    mode: str = "auto",
    tint_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
    threshold: float = 0.004,
) -> np.ndarray:
    """파일/HTTP가 모두 호출하는 단일 진입점.

    rgb: (H,W,3) uint8. RGBA는 호출 측에서 흰 배경 flatten 후 전달.
    """
    return multiply_to_alpha(rgb, mode=mode, tint_color=tint_color, bg_color=bg_color, threshold=threshold)


def clean_cut_one_touch(
    input_path: str,
    output_path: str,
    mode: str = "auto",
    threshold: float = 0.004,
    tint_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
) -> None:
    """파일 기반 진입점."""
    rgb = load_rgb(input_path)
    rgba = run_pipeline(rgb, mode=mode, tint_color=tint_color, bg_color=bg_color, threshold=threshold)
    save_png(rgba, output_path)


# ============================================================
# HTTP 서버 — 브라우저 원터치 인터페이스
# ============================================================

def _process_bytes(
    image_bytes: bytes,
    mode: str = "auto",
    tint_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
    threshold: float = 0.004,
) -> bytes:
    """HTTP 요청 처리 — 바이트 IO 후 run_pipeline 호출."""
    img = Image.open(BytesIO(image_bytes))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    rgb = np.array(img)

    rgba = run_pipeline(rgb, mode=mode, tint_color=tint_color, bg_color=bg_color, threshold=threshold)

    # 알파=0 픽셀의 RGB를 0으로 정리
    transparent = rgba[:, :, 3] == 0
    rgba[transparent, :3] = 0

    out = BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG")
    return out.getvalue()


def serve(port: int = 8766) -> None:
    """로컬 HTTP 서버 시작 — http://localhost:<port> 에서 원터치 UI 제공."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    def _strip_data_url(s: str) -> str:
        if not s:
            return ""
        return s.split(",", 1)[1] if "," in s else s

    def _parse_color(s, default):
        # "#RRGGBB" 또는 [r,g,b] 형태 허용
        if isinstance(s, (list, tuple)) and len(s) >= 3:
            return (int(s[0]), int(s[1]), int(s[2]))
        if isinstance(s, str) and s.startswith("#") and len(s) == 7:
            return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
        return default

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
                if not image_b64:
                    raise ValueError("image 필드 없음")
                image_bytes = base64.b64decode(image_b64)
                mode = str(data.get("mode", "auto"))
                tint = _parse_color(data.get("tint_color", "#000000"), (0, 0, 0))
                bg = _parse_color(data.get("bg_color", "#FFFFFF"), (255, 255, 255))
                threshold = float(data.get("threshold", 0.004))

                result_bytes = _process_bytes(
                    image_bytes,
                    mode=mode,
                    tint_color=tint,
                    bg_color=bg,
                    threshold=threshold,
                )

                # 자동 판별 결과를 헤더로도 알려줌
                detected = mode if mode != "auto" else auto_detect_mode(
                    np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
                )
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(err_msg.encode("utf-8"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("X-Detected-Mode", detected)
            self.send_header("Content-Length", str(len(result_bytes)))
            self.end_headers()
            self.wfile.write(result_bytes)

        def log_message(self, format, *args):
            sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    server = HTTPServer(("localhost", port), Handler)
    print(f"clean_cut_one_touch web — http://localhost:{port}")
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
        port = int(args[1]) if len(args) > 1 else 8766
        serve(port=port)
    elif len(args) >= 2:
        in_path = args[0]
        out_path = args[1]
        mode = args[2] if len(args) > 2 else "auto"
        thresh = float(args[3]) if len(args) > 3 else 0.004
        clean_cut_one_touch(in_path, out_path, mode=mode, threshold=thresh)
        print(f"완료: {out_path}")
    else:
        print("사용법:")
        print("  python clean_cut_one_touch.py serve [port]                       웹 서버 시작 (기본 8766)")
        print("  python clean_cut_one_touch.py <in> <out> [mode] [thresh]         파일 모드")
        print("    mode   : auto | grayscale | tint | color")
        print("    thresh : alpha 노이즈 컷오프 0~1 (기본 0.004)")
        sys.exit(1)
