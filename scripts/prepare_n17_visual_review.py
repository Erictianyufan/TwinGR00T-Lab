"""Build exhaustive numbered contact sheets; never approve a video review.

Requires the optional ``review`` extra. Only the manifest's same-stem MP4 is
read. A failed decode may leave partial sheets but never a completed index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

MAIN_CROP = (0, 60, 1280, 1020)
COLS, ROWS = 3, 4
THUMBNAIL = (640, 480)
CAPTION_HEIGHT = 40


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_manifest(manifest: dict) -> dict:
    if type(manifest.get("recording_schema_version")) is not int or manifest["recording_schema_version"] != 2:
        raise ValueError("Only recording_schema_version=2 has the required frame evidence")
    for key in ("steps_executed", "record_every", "video_frame_count", "expected_video_frames"):
        if type(manifest.get(key)) is not int or manifest[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("record_fps", "step_dt", "expected_video_duration_s"):
        if type(manifest.get(key)) not in (int, float) or not math.isfinite(manifest[key]) or manifest[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    expected = (manifest["steps_executed"] + manifest["record_every"] - 1) // manifest["record_every"]
    if manifest["video_frame_count"] != expected or manifest["expected_video_frames"] != expected:
        raise ValueError("Manifest frame counts disagree with executed control steps")
    if not math.isclose(manifest["record_fps"], 1 / manifest["step_dt"] / manifest["record_every"], rel_tol=1e-5):
        raise ValueError("Manifest recording FPS disagrees with control cadence")
    if not math.isclose(manifest["expected_video_duration_s"], expected / manifest["record_fps"], rel_tol=1e-5):
        raise ValueError("Manifest duration disagrees with frame count")
    digest = manifest.get("video_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Manifest has no valid video SHA256")
    return manifest


def prepare_review(manifest_path: Path, output_dir: Path, *, view: str = "full", reader=None) -> dict:
    """Stream every decoded frame once into lossless PNG sheets and an index."""
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if view not in ("full", "main"):
        raise ValueError("view must be full or main")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing review directory: {output_dir}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = validate_manifest(json.loads(manifest_bytes))
    video = manifest_path.with_suffix(".mp4")
    if sha256(video) != manifest["video_sha256"]:
        raise ValueError("Same-stem MP4 SHA256 does not match the manifest")
    if reader is None:
        from imageio_ffmpeg import read_frames

        reader = read_frames
    frames = reader(str(video), pix_fmt="rgb24", output_params=["-vsync", "0"])
    count = 0
    sheets = []
    current_frames = []
    sheet = None
    try:
        metadata = next(frames)
        width, height = metadata["size"]
        if any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError("Decoder returned invalid frame dimensions")
        if not math.isclose(float(metadata["fps"]), manifest["record_fps"], rel_tol=1e-3):
            raise ValueError("Decoded FPS disagrees with the manifest")
        duration = float(metadata["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Decoder returned no finite positive duration")
        if abs(duration - manifest["expected_video_duration_s"]) > max(0.02, 1 / manifest["record_fps"]):
            raise ValueError("Decoded duration disagrees with the manifest")
        if view == "main" and (width, height) != (1920, 1080):
            raise ValueError("Main-overview crop is defined only for the 1920x1080 recorder layout")
        output_dir.mkdir(parents=True, exist_ok=False)
        tile_width, tile_height = THUMBNAIL[0], THUMBNAIL[1] + CAPTION_HEIGHT
        for raw in frames:
            count += 1
            if count > manifest["video_frame_count"]:
                raise ValueError("Decoded more frames than the manifest")
            if len(raw) != width * height * 3:
                raise ValueError(f"Incomplete decoded RGB frame {count}")
            slot = (count - 1) % (COLS * ROWS)
            if slot == 0:
                sheet = Image.new("RGB", (COLS * tile_width, ROWS * tile_height), "white")
                current_frames = []
            frame = Image.frombytes("RGB", (width, height), raw)
            if view == "main":
                frame = frame.crop(MAIN_CROP)
            thumb = ImageOps.contain(frame, THUMBNAIL, method=Image.Resampling.LANCZOS)
            left, top = (slot % COLS) * tile_width, (slot // COLS) * tile_height
            sheet.paste(thumb, (left + (THUMBNAIL[0] - thumb.width) // 2, top + (THUMBNAIL[1] - thumb.height) // 2))
            step = 1 + (count - 1) * manifest["record_every"]
            frame_info = {
                "frame": count,
                "control_step": step,
                "video_time_s": (count - 1) / manifest["record_fps"],
                "simulation_time_s": step * manifest["step_dt"],
                "row": slot // COLS,
                "column": slot % COLS,
            }
            caption = (
                f"Frame {count:06d} | control step {step}\n"
                f"video {frame_info['video_time_s']:.3f}s | simulation {frame_info['simulation_time_s']:.3f}s"
            )
            ImageDraw.Draw(sheet).multiline_text((left + 6, top + THUMBNAIL[1] + 4), caption, fill="black", spacing=3)
            current_frames.append(frame_info)
            if slot == COLS * ROWS - 1 or count == manifest["video_frame_count"]:
                filename = f"sheet_{len(sheets) + 1:04d}.png"
                sheet.save(output_dir / filename)
                sheets.append(
                    {
                        "file": filename,
                        "first_frame": current_frames[0]["frame"],
                        "last_frame": count,
                        "frames": current_frames,
                    }
                )
        if count != manifest["video_frame_count"]:
            raise ValueError(f"Decoded {count} frames, expected {manifest['video_frame_count']}; incomplete video")
        if sha256(video) != manifest["video_sha256"] or manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Source video/manifest changed during review preparation")
    finally:
        close = getattr(frames, "close", None)
        if close:
            close()
    index = {
        "schema_version": 1,
        "status": "awaiting_visual_review",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_video": str(video),
        "source_video_sha256": manifest["video_sha256"],
        "source_video_review": manifest.get("video_review"),
        "source_acceptance_success": manifest.get("acceptance", {}).get("success"),
        "decoded_frame_count": count,
        "record_fps": manifest["record_fps"],
        "record_every": manifest["record_every"],
        "coverage": "every decoded frame in source order; no temporal sampling",
        "view": view,
        "crop_box_xyxy": list(MAIN_CROP) if view == "main" else None,
        "thumbnail_size": list(THUMBNAIL),
        "columns": COLS,
        "rows": ROWS,
        "review_limitations": (
            "Downscaled contact sheets are review aids, not physical-success or video-review approval. "
            "Main crop excludes the other camera panels; inspect original/all-camera evidence too."
        ),
        "sheets": sheets,
    }
    with (output_dir / "index.json").open("x", encoding="utf-8") as stream:
        json.dump(index, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("full", "main"), default="full")
    args = parser.parse_args()
    result = prepare_review(args.manifest, args.output_dir, view=args.view)
    print(
        json.dumps(
            {
                "status": result["status"],
                "frames": result["decoded_frame_count"],
                "sheets": len(result["sheets"]),
                "index": str(args.output_dir / "index.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
