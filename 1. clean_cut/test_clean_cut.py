"""clean_cut 테스트"""

import numpy as np
from PIL import Image, ImageDraw
import pytest

from clean_cut import (
    load_rgba,
    is_white,
    collect_edge_seeds,
    bfs_mark_background,
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


# ── collect_edge_seeds ──

def test_edge_seeds_all_white_border():
    """10x10 이미지, 테두리가 모두 흰색일 때 시작점 = 테두리 픽셀 수 (36개)"""
    pixels = np.full((10, 10, 4), 255, dtype=np.uint8)
    # 내부를 검정으로 채워도 테두리는 흰색
    pixels[1:9, 1:9, :3] = 0

    seeds = collect_edge_seeds(pixels)
    assert len(seeds) == 36  # 10*4 - 4(꼭짓점 중복 제거) = 36


# ── bfs_mark_background ──

def test_bfs_marks_background_not_object():
    """20x20 흰색 이미지 중앙에 5x5 검정 사각형 → 검정 영역은 마킹되지 않음"""
    pixels = np.full((20, 20, 4), 255, dtype=np.uint8)
    pixels[8:13, 8:13, :3] = 0  # 중앙 5x5 검정

    seeds = collect_edge_seeds(pixels)
    mask = bfs_mark_background(pixels, seeds)

    # 배경(흰색 영역)은 마킹됨
    assert mask[0, 0] is np.True_
    # 검정 사각형 내부는 마킹되지 않음
    assert mask[10, 10] is np.False_
    # 마킹된 개수 = 전체 - 검정 영역
    assert mask.sum() == 20 * 20 - 5 * 5


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
    """20x20 흰색 이미지 중앙에 5x5 검정 사각형 → 경계면 = 외곽 테두리 16개"""
    pixels = np.full((20, 20, 4), 255, dtype=np.uint8)
    pixels[8:13, 8:13, :3] = 0

    seeds = collect_edge_seeds(pixels)
    bg_mask = bfs_mark_background(pixels, seeds)
    boundary = find_boundary(bg_mask)

    # 경계면은 5x5 사각형의 외곽 테두리 = 5*4 - 4 = 16개
    assert boundary.sum() == 16
    # 내부 중심은 경계가 아님
    assert boundary[10, 10] is np.False_
    # 사각형 꼭짓점은 경계
    assert boundary[8, 8] is np.True_


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
    assert expanded[10, 10] is np.False_


# ── apply_boundary_alpha ──

def test_boundary_alpha_white_pixel():
    """흰색(255,255,255) 경계 픽셀 → 알파 ≈ 0"""
    pixels = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask)
    assert result[0, 0, 3] == 0


def test_boundary_alpha_gray_pixel():
    """회색(128,128,128) 경계 픽셀 → 알파 ≈ 128"""
    pixels = np.array([[[128, 128, 128, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask)
    assert 125 <= result[0, 0, 3] <= 131


def test_boundary_alpha_black_pixel():
    """검정(0,0,0) 경계 픽셀 → 알파 = 255"""
    pixels = np.array([[[0, 0, 0, 255]]], dtype=np.uint8)
    mask = np.array([[True]])
    result = apply_boundary_alpha(pixels, mask)
    assert result[0, 0, 3] == 255


# ── 통합 테스트 ──

def test_integration_black_circle_on_white(tmp_path):
    """흰 배경 위 검정 원 이미지 → 배경만 투명하게 된 PNG 출력"""
    # 테스트 이미지 생성: 흰 배경에 검정 원
    img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([15, 15, 35, 35], fill=(0, 0, 0, 255))

    in_path = str(tmp_path / "input.png")
    out_path = str(tmp_path / "output.png")
    img.save(in_path)

    clean_cut(in_path, out_path)

    # 출력 파일 검증
    result = Image.open(out_path)
    assert result.mode == "RGBA"

    result_np = np.array(result)

    # 모서리(배경)는 투명해야 함
    assert result_np[0, 0, 3] == 0
    # 원 중심은 불투명해야 함
    assert result_np[25, 25, 3] == 255
