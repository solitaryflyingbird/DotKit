"""
idle_motion — 도트 스프라이트 idle 진동 애니메이션 생성 도구

투명 배경 PNG에서 ±1px Y축 진동 4프레임 idle 애니메이션을 생성한다.
"""

from PIL import Image
import os
import sys
import json


def load_image(path: str) -> Image.Image:
    """이미지를 RGBA로 로드하고 불투명 영역만 크롭한다."""
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def make_idle(input_path: str, output_path: str,
              num_frames: int = 4, duration: int = 300,
              y_offsets: list = None) -> None:
    """idle 진동 애니메이션을 생성한다.

    Args:
        input_path: 입력 이미지 경로 (RGBA PNG)
        output_path: 출력 GIF 경로
        num_frames: 프레임 수 (기본 4)
        duration: 프레임당 ms (기본 300)
        y_offsets: 프레임별 Y 오프셋 리스트 (기본 [0, -1, 0, 1])
    """
    src = load_image(input_path)
    w, h = src.size

    if y_offsets is None:
        y_offsets = [0, -1, 0, 1]

    # 캔버스: 상하 1px 여백
    pad = 1
    canvas_w = w
    canvas_h = h + pad * 2

    frames = []
    for i in range(num_frames):
        frame = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        paste_y = pad + y_offsets[i]
        frame.paste(src, (0, paste_y), src)
        frames.append(frame)

    # GIF 저장 (회색 배경 합성)
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

    # 스프라이트시트 PNG 저장
    sheet_path = output_path.rsplit(".", 1)[0] + "_sheet.png"
    sheet = Image.new("RGBA", (canvas_w * num_frames, canvas_h), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, (canvas_w * i, 0), f)
    sheet.save(sheet_path)

    print(f"완료: {output_path}")
    print(f"시트: {sheet_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python idle_motion.py <입력 PNG> <출력 GIF> [설정 JSON]")
        print()
        print("설정 JSON 예시:")
        print(json.dumps({
            "num_frames": 4,
            "duration": 300,
            "y_offsets": [0, -1, 0, 1]
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

    make_idle(in_path, out_path, **kwargs)
