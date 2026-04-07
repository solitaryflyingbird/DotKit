"""
clean_cut — 스프라이트 선따기 도구

흰색 배경을 가장자리 BFS로 탐색하여 제거하고,
경계면 픽셀을 알파 블렌딩하여 깔끔하게 오려낸다.
"""

from collections import deque
from PIL import Image
import numpy as np


def load_rgba(path: str) -> np.ndarray:
    """이미지를 RGBA numpy 배열로 로드한다."""
    img = Image.open(path).convert("RGBA")
    return np.array(img)


def is_white(pixel_rgb, threshold: int = 10) -> bool:
    """RGB 값이 흰색에 충분히 가까운지 판별한다."""
    r, g, b = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])
    return (255 - r) <= threshold and (255 - g) <= threshold and (255 - b) <= threshold


def collect_edge_seeds(pixels: np.ndarray, threshold: int = 10) -> list[tuple[int, int]]:
    """이미지 가장자리(테두리)에서 흰색인 픽셀 좌표를 시작점으로 수집한다."""
    h, w = pixels.shape[:2]
    seeds = []
    for y in range(h):
        for x in range(w):
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                if is_white(pixels[y, x, :3], threshold):
                    seeds.append((y, x))
    return seeds


def bfs_mark_background(pixels: np.ndarray, seeds: list[tuple[int, int]], threshold: int = 10) -> np.ndarray:
    """BFS로 시작점에서 인접 흰색 픽셀을 확장하여 배경 영역을 마킹한다.

    Returns:
        bool 배열 (H x W). True이면 배경.
    """
    h, w = pixels.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    for y, x in seeds:
        if not visited[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if is_white(pixels[ny, nx, :3], threshold):
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    return visited


def remove_background(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """배경으로 마킹된 픽셀의 알파를 0으로 설정한다."""
    result = pixels.copy()
    result[mask, 3] = 0
    return result


def find_boundary(bg_mask: np.ndarray) -> np.ndarray:
    """오브젝트(비배경) 픽셀 중 배경과 인접한 픽셀을 경계면으로 식별한다.

    Returns:
        bool 배열 (H x W). True이면 경계면.
    """
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
    """경계면에서 오브젝트 안쪽으로 depth 픽셀만큼 확장한다.

    Returns:
        bool 배열 (H x W). True이면 확장된 경계면 영역.
    """
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


def apply_boundary_alpha(pixels: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    """경계면 픽셀의 밝기(흰색 근접도)에 반비례하여 알파를 매핑한다.

    흰색에 가까울수록 투명, 오브젝트 본래 색에 가까울수록 불투명.
    """
    result = pixels.copy()
    ys, xs = np.where(boundary_mask)

    r = result[ys, xs, 0].astype(float)
    g = result[ys, xs, 1].astype(float)
    b = result[ys, xs, 2].astype(float)

    whiteness = (r + g + b) / (3.0 * 255.0)
    alpha = ((1.0 - whiteness) * 255.0).clip(0, 255).astype(np.uint8)

    result[ys, xs, 3] = alpha
    return result


def save_png(pixels: np.ndarray, path: str) -> None:
    """RGBA numpy 배열을 PNG로 저장한다."""
    img = Image.fromarray(pixels, "RGBA")
    img.save(path)


def save_psd(pixels: np.ndarray, path: str) -> None:
    """RGBA 결과를 PSD로 저장한다. 확인용 배경 레이어 포함."""
    from psd_tools import PSDImage

    h, w = pixels.shape[:2]
    psd = PSDImage.new("RGBA", (w, h))

    # 불투명 픽셀의 평균색 → 보색 계산
    opaque = pixels[:, :, 3] > 0
    if np.any(opaque):
        avg_r = np.mean(pixels[opaque, 0])
        avg_g = np.mean(pixels[opaque, 1])
        avg_b = np.mean(pixels[opaque, 2])
    else:
        avg_r, avg_g, avg_b = 128, 128, 128
    comp_r, comp_g, comp_b = 255 - int(avg_r), 255 - int(avg_g), 255 - int(avg_b)

    # 확인용 배경: 보색 단색
    bg = np.full((h, w, 4), 255, dtype=np.uint8)
    bg[:, :, 0] = comp_r
    bg[:, :, 1] = comp_g
    bg[:, :, 2] = comp_b

    # 레이어 순서: cut(위) → bg(아래)
    # PSD append 순서 = 위에서 아래로
    cut_layer = psd.create_pixel_layer(
        name="cut", image=Image.fromarray(pixels, "RGBA"))
    bg_layer = psd.create_pixel_layer(
        name="bg", image=Image.fromarray(bg, "RGBA"))

    psd.append(bg_layer)
    psd.append(cut_layer)
    psd.save(path)


def clean_cut(input_path: str, output_path: str, threshold: int = 10, boundary_depth: int = 1) -> None:
    """스프라이트 추출 파이프라인.

    1. 이미지 로드
    2. 가장자리 흰색 시작점 수집
    3. BFS 배경 마킹
    4. 배경 알파 0 처리
    5. 경계면 감지 및 알파 처리
    6. 출력 — 확장자가 .psd면 확인용 배경 레이어 포함 PSD, 그 외엔 PNG
    """
    pixels = load_rgba(input_path)
    seeds = collect_edge_seeds(pixels, threshold)
    bg_mask = bfs_mark_background(pixels, seeds, threshold)
    result = remove_background(pixels, bg_mask)
    boundary = find_boundary(bg_mask)
    expanded = expand_boundary(boundary, bg_mask, depth=boundary_depth)
    result = apply_boundary_alpha(result, expanded)

    if output_path.lower().endswith(".psd"):
        save_psd(result, output_path)
    else:
        save_png(result, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("사용법: python clean_cut.py <입력 이미지> <출력 PSD> [threshold]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    thresh = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    clean_cut(in_path, out_path, thresh)
    print(f"완료: {out_path}")
