"""Render a labeled four-camera RoboTwin demonstration as a 1080p MP4."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 31.25
CANVAS_SIZE = (1920, 1080)
HEADER_HEIGHT = 60
MAIN_SIZE = (1280, 960)
SIDE_SIZE = (640, 320)
CAMERAS = (
    ("front_camera_rgb", "FRONT · OVERALL ROBOT + OBJECTS"),
    ("head_camera_rgb", "HEAD"),
    ("left_camera_rgb", "LEFT WRIST"),
    ("right_camera_rgb", "RIGHT WRIST"),
)


def encoded_frame(value: Any) -> Image.Image:
    payload = bytes(value) if isinstance(value, (bytes, np.bytes_)) else np.asarray(value).tobytes()
    with Image.open(BytesIO(payload)) as image:
        return image.convert("RGB")


def letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, (12, 15, 20))
    source = image.copy()
    source.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - source.width) // 2
    y = (size[1] - source.height) // 2
    result.paste(source, (x, y))
    return result


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def draw_panel_label(canvas: Image.Image, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    x, y = xy
    box = draw.textbbox((x + 14, y + 12), text, font=font)
    draw.rounded_rectangle((x + 8, y + 7, box[2] + 9, box[3] + 8), radius=7, fill=(0, 0, 0, 175))
    draw.text((x + 14, y + 12), text, font=font, fill=(255, 255, 255, 255))


def instruction_for(demo: Path) -> str:
    instruction_path = demo.parent.parent / "instructions" / f"{demo.stem}.json"
    if not instruction_path.exists():
        return "stack the three bowls"
    payload = json.loads(instruction_path.read_text(encoding="utf-8"))
    for key in ("seen", "unseen"):
        values = payload.get(key, [])
        if values:
            return str(values[0])
    return "stack the three bowls"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=FPS)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    task = instruction_for(args.demo)
    title_font = load_font(28, bold=True)
    label_font = load_font(20, bold=True)
    time_font = load_font(18)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}",
        "-framerate",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]

    with h5py.File(args.demo, "r") as handle:
        group = handle["observations/camera_obs"]
        missing = [key for key, _ in CAMERAS if key not in group]
        if missing:
            raise ValueError(f"Missing cameras: {missing}")
        frames = min(len(group[key]) for key, _ in CAMERAS)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdin is not None
        try:
            for index in range(frames):
                canvas = Image.new("RGB", CANVAS_SIZE, (6, 9, 14))
                front = letterbox(encoded_frame(group[CAMERAS[0][0]][index]), MAIN_SIZE)
                canvas.paste(front, (0, HEADER_HEIGHT))
                draw_panel_label(canvas, (0, HEADER_HEIGHT), CAMERAS[0][1], label_font)

                for panel_index, (camera_key, label) in enumerate(CAMERAS[1:]):
                    y = HEADER_HEIGHT + panel_index * SIDE_SIZE[1]
                    panel = letterbox(encoded_frame(group[camera_key][index]), SIDE_SIZE)
                    canvas.paste(panel, (MAIN_SIZE[0], y))
                    draw_panel_label(canvas, (MAIN_SIZE[0], y), label, label_font)

                draw = ImageDraw.Draw(canvas)
                header = f"TwinGR00T Lab  |  stack_bowls_three  |  {args.demo.stem}"
                draw.text((22, 13), header, font=title_font, fill=(239, 244, 255))
                timestamp = f"t = {index / args.fps:06.2f} s  |  {task}"
                draw.text((1295, 1037), timestamp, font=time_font, fill=(190, 205, 225))
                process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
                if index == 0 or (index + 1) % 100 == 0 or index + 1 == frames:
                    print(f"[render] {index + 1}/{frames}", flush=True)
        finally:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")

    manifest = {
        "status": "pass",
        "demo": str(args.demo),
        "output": str(args.output),
        "frames": frames,
        "fps": args.fps,
        "duration_seconds": frames / args.fps,
        "resolution": list(CANVAS_SIZE),
        "cameras": [key for key, _ in CAMERAS],
        "task": task,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
