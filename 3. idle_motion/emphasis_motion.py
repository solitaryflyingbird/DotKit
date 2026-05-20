"""
emphasis_motion — 마스크 영역 국소 bulge 8컷 애니메이션 (per-region profile)

입력:
- IN_PATH   : 원본 캐릭터 RGBA
- MASK_PATH : 강조할 영역을 빨강으로 칠한 마스크 (반투명 OK)

영역별 프로파일:
  Region 0 (가슴): 단일 sub-bulge, 수평만 (ky=0, kx 부스트)
  Region 1 (허벅지): 3개 sub-bulge
    - 중앙(사타구니) isotropic (kx=1, ky=1)
    - 좌측 outer (offset x=-0.6·half_w), 수평만
    - 우측 outer (offset x=+0.6·half_w), 수평만

각 sub-bulge의 변위 (단일 bilinear 샘플):
  dx = (x − sub_cx) · k · exp(−((x − sub_cx)² + (y − sub_cy)²) / (2σ_sub²)) · kx
  dy = (y − sub_cy) · k · exp(−((x − sub_cx)² + (y − sub_cy)²) / (2σ_sub²)) · ky
  k  = amp / (σ_sub · e^(−0.5))   → 피크 변위 = amp at r=σ_sub

모든 sub-bulge 변위를 단순 합산 → 한 번의 bilinear 샘플 → 잔상 없음.
"""
import os
import json
import math
import numpy as np
from PIL import Image

IN_PATH = "/Users/js/Desktop/도트킷/예시 이미지들 커밋제외/1.png"
MASK_PATH = "/Users/js/Desktop/도트킷/생성예시파일 커밋제외/마스킹.png"
OUT_DIR = "/Users/js/Desktop/도트킷/생성예시파일 커밋제외/"

N_FRAMES = 8
MAX_AMP = 12          # 피크 변위 기본 (px)
BULGE_SIGMA = 60      # 기본 σ (px)
MIN_REGION_PX = 10000

RED_DIFF_MIN = 40
ALPHA_MIN = 50

# 영역별 sub-bulge 프로파일
#   ox_frac / oy_frac : 영역 중심에서의 offset (half_w / half_h 대비 비율)
#   kx / ky           : x/y 방향 진폭 스케일 (0 = 해당 축 이동 없음)
#   sigma_scale       : σ 배율
REGION_PROFILES = [
    # Region 0: 가슴 — 수평 주, 수직 약간 (자연스러운 부풀기)
    [
        {"ox_frac": 0.0, "oy_frac": 0.0, "kx": 0.7, "ky": 0.35, "sigma_scale": 1.0,
         "note": "chest bulge (horizontal dominant, slight vertical)"},
    ],
    # Region 1: 허벅지 — 사타구니 isotropic + 양옆 수평
    [
        {"ox_frac": 0.0,  "oy_frac": 0.0, "kx": 1.0, "ky": 1.0, "sigma_scale": 1.0,
         "note": "crotch isotropic"},
        {"ox_frac": -0.6, "oy_frac": 0.0, "kx": 1.2, "ky": 0.0, "sigma_scale": 0.8,
         "note": "left outer thigh horizontal"},
        {"ox_frac": 0.6,  "oy_frac": 0.0, "kx": 1.2, "ky": 0.0, "sigma_scale": 0.8,
         "note": "right outer thigh horizontal"},
    ],
]

# 프로파일 밖 영역에 대한 기본값 (등방성 radial bulge)
DEFAULT_PROFILE = [
    {"ox_frac": 0.0, "oy_frac": 0.0, "kx": 1.0, "ky": 1.0, "sigma_scale": 1.0, "note": "default"},
]

# 마스크에 없는 수동 sub-bulge (허리 수축 등)
# cy는 가슴/허벅지 region의 중간에 자동 배치, cx는 두 region cx 평균
MANUAL_BULGES = [
    {
        "cx_source": "midline",   # 두 region cx 평균
        "cy_source": "between",   # 두 region cy 중간
        "sigma": 50,
        "kx": -0.35,              # 음수 = 수축 (몸 중심으로 당김)
        "ky": 0.0,
        "note": "waist horizontal contraction",
    },
]


# ------------------------------------------------------------
# image helpers
# ------------------------------------------------------------
def load_rgba(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def bilinear_sample(src_f: np.ndarray, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    h, w = src_f.shape[:2]
    xi_c = np.clip(xi, 0, w - 1)
    yi_c = np.clip(yi, 0, h - 1)
    x0 = np.floor(xi_c).astype(np.int32)
    y0 = np.floor(yi_c).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xi_c - x0).astype(np.float32)[:, :, None]
    fy = (yi_c - y0).astype(np.float32)[:, :, None]
    tl = src_f[y0, x0]
    tr = src_f[y0, x1]
    bl = src_f[y1, x0]
    br = src_f[y1, x1]
    top = tl * (1 - fx) + tr * fx
    bot = bl * (1 - fx) + br * fx
    return top * (1 - fy) + bot * fy


# ------------------------------------------------------------
# mask region detection
# ------------------------------------------------------------
def extract_red(mask_np: np.ndarray) -> np.ndarray:
    r = mask_np[:, :, 0].astype(np.int32)
    g = mask_np[:, :, 1].astype(np.int32)
    b = mask_np[:, :, 2].astype(np.int32)
    a = mask_np[:, :, 3].astype(np.int32)
    return (r - np.maximum(g, b) > RED_DIFF_MIN) & (a > ALPHA_MIN)


def row_connected_groups(red: np.ndarray):
    h = red.shape[0]
    row_has = red.any(axis=1)
    out = []
    cur = None
    for y in range(h):
        if row_has[y]:
            if cur is None:
                cur = [y, y]
            else:
                cur[1] = y
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def detect_regions(mask_np: np.ndarray):
    red = extract_red(mask_np)
    groups = row_connected_groups(red)
    regions = []
    for g0, g1 in groups:
        sub = red[g0:g1 + 1]
        cnt = int(sub.sum())
        if cnt < MIN_REGION_PX:
            continue
        ys, xs = np.where(sub)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min() + g0), int(ys.max() + g0)
        regions.append({
            "y_range": (g0, g1),
            "bbox": (x_min, y_min, x_max, y_max),
            "half_w": (x_max - x_min) / 2.0,
            "half_h": (y_max - y_min) / 2.0,
            "pixel_count": cnt,
            "cx": float(xs.mean()),
            "cy": float(ys.mean() + g0),
        })
    return regions


# ------------------------------------------------------------
# sub-bulge flattening (mask-detected + manual)
# ------------------------------------------------------------
def gather_sub_bulges(regions, base_sigma: float):
    """모든 sub-bulge를 절대 좌표 리스트로 평탄화."""
    subs = []
    for r_idx, region in enumerate(regions):
        profile = REGION_PROFILES[r_idx] if r_idx < len(REGION_PROFILES) else DEFAULT_PROFILE
        for p in profile:
            subs.append({
                "cx": region["cx"] + p["ox_frac"] * region["half_w"],
                "cy": region["cy"] + p["oy_frac"] * region["half_h"],
                "sigma": base_sigma * p["sigma_scale"],
                "kx": p["kx"],
                "ky": p["ky"],
                "note": f"R{r_idx}: {p['note']}",
            })

    # 수동 sub-bulge (허리 등)
    if len(regions) >= 2:
        r0, r1 = regions[0], regions[1]
        mid_cx = (r0["cx"] + r1["cx"]) / 2
        mid_cy = (r0["cy"] + r1["cy"]) / 2
        for m in MANUAL_BULGES:
            mcx = mid_cx if m["cx_source"] == "midline" else m.get("cx", mid_cx)
            mcy = mid_cy if m["cy_source"] == "between" else m.get("cy", mid_cy)
            subs.append({
                "cx": mcx,
                "cy": mcy,
                "sigma": m["sigma"],
                "kx": m["kx"],
                "ky": m["ky"],
                "note": f"manual: {m['note']}",
            })
    return subs


# ------------------------------------------------------------
# displacement field
# ------------------------------------------------------------
def compute_displacement(h: int, w: int, sub_bulges, amp: float):
    x_grid, y_grid = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    dx_total = np.zeros((h, w), dtype=np.float32)
    dy_total = np.zeros((h, w), dtype=np.float32)
    if amp == 0 or not sub_bulges:
        return dx_total, dy_total

    for sub in sub_bulges:
        sigma = sub["sigma"]
        k = amp / (sigma * math.exp(-0.5))
        dxn = x_grid - sub["cx"]
        dyn = y_grid - sub["cy"]
        r2 = dxn * dxn + dyn * dyn
        falloff = np.exp(-r2 / (2 * sigma * sigma))
        dx_total += dxn * k * falloff * sub["kx"]
        dy_total += dyn * k * falloff * sub["ky"]

    return dx_total, dy_total


def warp_frame(src_f: np.ndarray, sub_bulges, amp: float) -> np.ndarray:
    if amp == 0:
        return src_f.copy()
    h, w = src_f.shape[:2]
    dx, dy = compute_displacement(h, w, sub_bulges, amp)
    x_grid, y_grid = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    xi = x_grid - dx
    yi = y_grid - dy
    return bilinear_sample(src_f, xi, yi)


# ------------------------------------------------------------
# frame generation
# ------------------------------------------------------------
def build_frames(src_np: np.ndarray, regions):
    src_f = src_np.astype(np.float32)
    sub_bulges = gather_sub_bulges(regions, BULGE_SIGMA)

    frames, logs = [], []
    for i in range(N_FRAMES):
        t = (1 - math.cos(2 * math.pi * i / N_FRAMES)) / 2
        amp = MAX_AMP * t
        out_f = warp_frame(src_f, sub_bulges, amp)
        out_u8 = np.clip(out_f, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(out_u8, "RGBA"))

        sub_details = [{
            "note": s["note"],
            "center": [round(s["cx"], 1), round(s["cy"], 1)],
            "sigma": round(s["sigma"], 1),
            "kx": s["kx"],
            "ky": s["ky"],
            "peak_px_x": round(amp * s["kx"], 2),
            "peak_px_y": round(amp * s["ky"], 2),
        } for s in sub_bulges]
        logs.append({
            "frame": i,
            "t": round(t, 4),
            "amp_base_px": round(amp, 3),
            "sub_bulges": sub_details,
        })
    return frames, logs


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src_np = load_rgba(IN_PATH)
    mask_np = load_rgba(MASK_PATH)
    if src_np.shape[:2] != mask_np.shape[:2]:
        raise ValueError(f"크기 불일치: {src_np.shape[:2]} vs {mask_np.shape[:2]}")
    h, w = src_np.shape[:2]

    regions = detect_regions(mask_np)
    print(f"탐지된 영역 {len(regions)}개:")
    for i, r in enumerate(regions):
        print(f"  #{i}: y={r['y_range']}, {r['pixel_count']}px, center=({r['cx']:.1f}, {r['cy']:.1f})")

    sub_bulges = gather_sub_bulges(regions, BULGE_SIGMA)
    print(f"\nsub-bulge {len(sub_bulges)}개:")
    for s in sub_bulges:
        print(f"  {s['note']:50s}: ({s['cx']:.1f}, {s['cy']:.1f}), σ={s['sigma']:.1f}, "
              f"kx={s['kx']:+.2f}, ky={s['ky']:+.2f}")

    frames, logs = build_frames(src_np, regions)

    for i, f in enumerate(frames):
        f.save(os.path.join(OUT_DIR, f"emphasis_{i}.png"))

    sheet = Image.new("RGBA", (w * N_FRAMES, h), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, (w * i, 0), f)
    sheet.save(os.path.join(OUT_DIR, "emphasis_sheet.png"))

    gif_frames = []
    for f in frames:
        bg = Image.new("RGBA", f.size, (220, 220, 220, 255))
        bg.paste(f, (0, 0), f)
        gif_frames.append(bg.convert("P", palette=Image.Palette.ADAPTIVE))
    gif_frames[0].save(
        os.path.join(OUT_DIR, "emphasis.gif"),
        save_all=True,
        append_images=gif_frames[1:],
        duration=150,
        loop=0,
        disposal=2,
    )

    meta = {
        "source": IN_PATH,
        "mask": MASK_PATH,
        "size": [w, h],
        "bulge_sigma_px": BULGE_SIGMA,
        "max_amp_px": MAX_AMP,
        "n_frames": N_FRAMES,
        "regions": [
            {"index": i, "y_range": list(r["y_range"]), "bbox": list(r["bbox"]),
             "pixel_count": r["pixel_count"], "cx": r["cx"], "cy": r["cy"],
             "half_w": r["half_w"], "half_h": r["half_h"]}
            for i, r in enumerate(regions)
        ],
        "region_profiles": REGION_PROFILES,
        "manual_bulges": MANUAL_BULGES,
        "operation_definitions": {
            "SUB_BULGE(sub_cx, sub_cy, σ, kx, ky)": (
                "dx = (x − sub_cx) · k · exp(−((x − sub_cx)² + (y − sub_cy)²) / (2σ²)) · kx, "
                "dy = 같은 방식으로 · ky. "
                "k = amp / (σ · e^(−0.5)) → r=σ에서 피크 변위 = amp. "
                "kx/ky=0이면 해당 축 이동 없음."
            ),
            "REGION_WARP": (
                "영역별 sub-bulge 변위장 합산 후 단일 bilinear 샘플. "
                "xi = x − Σ dx, yi = y − Σ dy, output = bilinear_sample(src, xi, yi)."
            ),
        },
        "frames": logs,
    }
    with open(os.path.join(OUT_DIR, "emphasis_log.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"완료: {OUT_DIR}emphasis.gif")


if __name__ == "__main__":
    main()
