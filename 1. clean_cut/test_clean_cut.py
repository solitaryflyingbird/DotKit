"""clean_cut 테스트"""

import numpy as np
from PIL import Image, ImageDraw
import pytest

from clean_cut import (
    load_rgba,
    is_white,
    mark_background_from_mask,
    remove_background,
    find_boundary,
    expand_boundary,
    apply_boundary_alpha,
    save_png,
    clean_cut,
)


# ── is_white ──

def test_is_white_pure_white():
    assert is_white((255, 255, 255)) is True


def test_is_white_near_white():
    assert is_white((250, 251, 253), threshold=10) is True


def test_is_white_gray():
    assert is_white((200, 200, 200), threshold=10) is False


# ── load_rgba ──

def test_load_rgba_white_image(tmp_path):
    """10x10 흰색 PNG 로드 시 모든 픽셀이 (255,255,255,255)"""
    img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
    path = tmp_path / "white.png"
    img.save(path)

    pixels = load_rgba(str(path))
    assert pixels.shape == (10, 10, 4)
    assert np.all(pixels == 255)


# ── mark_background_from_mask ──

def test_mask_floods_white_region():
    """20x20 흰 배경 + 중앙 5x5 검정 사각형, 마스크 모서리 한 점에서 BFS → 흰 영역 전체 채워짐"""
    pixels = np.full((20, 20, 4), 255, dtype=np.uint8)
    pixels[8:13, 8:13, :3] = 0  # 중앙 5x5 검정

    # 마스크: (0,0)에 알파 있는 한 점
    mask_pixels = np.zeros((20, 20, 4), dtype=np.uint8)
    mask_pixels[0, 0] = [255, 0, 0, 255]  # 빨강 ROI 한 점

    bg = np.zeros((20, 20), dtype=bool)
    bg = mark_background_from_mask(pixels, bg, mask_pixels)

    # 배경(흰색 영역)은 마킹됨
    assert bg[0, 0]
    # 검정 사각형 내부는 마킹되지 않음
    assert not bg[10, 10]
    # 마킹된 개수 = 전체 - 검정 영역
    assert bg.sum() == 20 * 20 - 5 * 5


def test_mask_no_seed_no_marking():
    """마스크가 비어있으면 bg_mask는 변화 없음 (no-op)"""
    pixels = np.full((20, 20, 4), 255, dtype=np.uint8)
    mask_pixels = np.zeros((20, 20, 4), dtype=np.uint8)  # 전부 알파 0
    bg = np.zeros((20, 20), dtype=bool)
    bg = mark_background_from_mask(pixels, bg, mask_pixels)
    assert bg.sum() == 0


# ── remove_background ──

def test_remove_background_alpha():
    """마킹된 영역의 알파=0, 마킹되지 않은 영역의 알파=255"""
    pixels = np.full((10, 10, 4), 255, dtype=np.uint8)
    pixels[3:7, 3:7, :3] = 0  # 중앙 4x4 검정

    mask = np.full((10, 10), True)
    mask[3:7, 3:7] = False

    result = remove_background(pixels, mask)

    assert np.all(result[mask, 3] == 0)
    assert np.all(result[~mask, 3] == 255)


# ── find_boundary ──

def test_find_boundary():
    """20x20 bg에 5x5 obj 영역 → 경계면 = obj 외곽 테두리 16개"""
    bg_mask = np.full((20, 20), True)
    bg_mask[8:13, 8:13] = False  # 중앙 5x5가 오브젝트
    boundary = find_boundary(bg_mask)

    # 경계면은 5x5 사각형의 외곽 테두리 = 5*4 - 4 = 16개
    assert boundary.sum() == 16
    # 내부 중심은 경계가 아님
    assert not boundary[10, 10]
    # 사각형 꼭짓점은 경계
    assert boundary[8, 8]


# ── expand_boundary ──

def test_expand_boundary_depth():
    """depth=1일 때 경계면 + 안쪽 1픽셀 레이어까지 포함"""
    bg_mask = np.full((20, 20), True)
    bg_mask[5:15, 5:15] = False  # 10x10 오브젝트

    boundary = find_boundary(bg_mask)
    expanded = expand_boundary(boundary, bg_mask, depth=1)

    # 원래 경계(외곽 36개) + 안쪽 1층(28개) = 64개
    assert expanded.sum() > boundary.sum()
    # 오브젝트 중심은 포함되지 않음
    assert not expanded[10, 10]


# ── apply_boundary_alpha ──

def test_boundary_alpha_white_pixel():
    """흰색(255,255,255) 경계 픽셀 → RGB=(0,0,0), alpha=0"""
    pixels = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask)
    assert result[0, 0, 3] == 0
    assert result[0, 0, 0] == 0 and result[0, 0, 1] == 0 and result[0, 0, 2] == 0


def test_boundary_alpha_midgray_pixel():
    """중간 회색(128,128,128) → RGB=(0,0,0), alpha = (1-0.502)*255 ≈ 127"""
    pixels = np.array([[[128, 128, 128, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask)
    assert 125 <= result[0, 0, 3] <= 129
    assert tuple(result[0, 0, :3]) == (0, 0, 0)


def test_boundary_alpha_dark_pixel():
    """짙은 회색(40,40,40) → RGB=(0,0,0), alpha = (1-0.157)*255 ≈ 215"""
    pixels = np.array([[[40, 40, 40, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask)
    assert 213 <= result[0, 0, 3] <= 217
    assert tuple(result[0, 0, :3]) == (0, 0, 0)


def test_boundary_alpha_bright_boundary():
    """밝은 경계(230,230,230) → alpha = (1-0.902)*255 ≈ 25"""
    pixels = np.array([[[230, 230, 230, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask)
    assert 23 <= result[0, 0, 3] <= 27
    assert tuple(result[0, 0, :3]) == (0, 0, 0)


def test_boundary_alpha_black_pixel():
    """검정(0,0,0) → alpha = 255 (이미 검정이라 변화 없음)"""
    pixels = np.array([[[0, 0, 0, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask)
    assert result[0, 0, 3] == 255
    assert tuple(result[0, 0, :3]) == (0, 0, 0)


def test_boundary_alpha_low_strength():
    """strength=0 → threshold=1.0 → 모든 픽셀 원본 유지 (RGB도 보존)"""
    pixels = np.array([[[230, 230, 230, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask, strength=0)
    assert result[0, 0, 3] == 255
    assert tuple(result[0, 0, :3]) == (230, 230, 230)


def test_boundary_alpha_max_strength():
    """strength=200 → threshold=-1.0 → 전환 구간 무시, 완전 적용"""
    pixels = np.array([[[40, 40, 40, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask, strength=200)
    assert 213 <= result[0, 0, 3] <= 217
    assert tuple(result[0, 0, :3]) == (0, 0, 0)


# ── 통합 테스트 ──

def test_integration_black_circle_on_white(tmp_path):
    """흰 배경 + 검정 원 + 모서리 마스크 ROI → 배경만 투명, 원은 보존"""
    # 원본: 흰 배경에 검정 원
    img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([15, 15, 35, 35], fill=(0, 0, 0, 255))

    # 마스크: 모서리에 ROI 한 점 (알파 기반)
    mask = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    mask.putpixel((0, 0), (255, 0, 0, 255))

    in_path = str(tmp_path / "input.png")
    mask_path = str(tmp_path / "mask.png")
    out_path = str(tmp_path / "output.png")
    img.save(in_path)
    mask.save(mask_path)

    clean_cut(in_path, out_path, mask_path=mask_path)

    result = Image.open(out_path)
    assert result.mode == "RGBA"
    result_np = np.array(result)

    # 모서리(배경)는 투명해야 함
    assert result_np[0, 0, 3] == 0
    # 원 중심은 불투명해야 함
    assert result_np[25, 25, 3] == 255


def test_integration_no_mask_is_noop(tmp_path):
    """마스크가 없으면 결과 이미지의 알파는 원본 그대로 유지 (no-op)"""
    img = Image.new("RGBA", (20, 20), (255, 255, 255, 255))
    img.putpixel((10, 10), (0, 0, 0, 255))
    in_path = str(tmp_path / "input.png")
    out_path = str(tmp_path / "output.png")
    img.save(in_path)

    clean_cut(in_path, out_path)  # mask_path 없음

    result_np = np.array(Image.open(out_path))
    # 모든 픽셀의 알파가 그대로 255
    assert np.all(result_np[..., 3] == 255)
