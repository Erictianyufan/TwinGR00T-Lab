"""Estimate RoboTwin colored-block centers from a demonstration's first RGB frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

COLOR_RANGES = {
    "block1": ((0, 105, 70), (12, 255, 255)),
    "block2": ((38, 70, 45), (92, 255, 255)),
    "block3": ((96, 75, 45), (132, 255, 255)),
}


def decode_rgb(value: object) -> np.ndarray:
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode the first RGB frame")
    # RoboTwin intentionally passes source RGB arrays directly to cv2.imencode.
    # cv2.imdecode therefore returns BGR channel values that numerically match
    # the original RGB array; do not swap them here.
    return decoded


def largest_component_centroid(mask: np.ndarray, minimum_area: int = 20) -> tuple[float, float]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("No foreground component found")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    if area < minimum_area:
        raise ValueError(f"Largest component is too small: {area} pixels")
    u, v = centroids[component]
    return float(u), float(v)


def pixel_to_world_on_plane(
    u: float,
    v: float,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    camera_origin_world = -rotation.T @ translation
    ray_camera = np.linalg.inv(intrinsic) @ np.array([u, v, 1.0], dtype=np.float64)
    ray_world = rotation.T @ ray_camera
    if abs(float(ray_world[2])) < 1e-8:
        raise ValueError("Camera ray is parallel to the table plane")
    distance = (plane_z - camera_origin_world[2]) / ray_world[2]
    return camera_origin_world + distance * ray_world


def estimate(
    path: Path, camera: str, plane_z: float, block_count: int = 3
) -> tuple[dict[str, list[float]], np.ndarray, dict[str, tuple[float, float]]]:
    with h5py.File(path, "r") as handle:
        group = handle[f"observation/{camera}"]
        image = decode_rgb(group["rgb"][0])
        intrinsic = np.asarray(group["intrinsic_cv"][0], dtype=np.float64)
        extrinsic = np.asarray(group["extrinsic_cv"][0], dtype=np.float64)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    positions: dict[str, list[float]] = {}
    pixels: dict[str, tuple[float, float]] = {}
    for name, (lower, upper) in list(COLOR_RANGES.items())[:block_count]:
        mask = cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        u, v = largest_component_centroid(mask)
        world = pixel_to_world_on_plane(u, v, intrinsic, extrinsic, plane_z)
        positions[name] = [round(float(value), 6) for value in world]
        pixels[name] = (u, v)
    return positions, image, pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=Path)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--plane-z", type=float, default=0.766)
    parser.add_argument("--blocks", type=int, choices=(2, 3), default=3)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()

    positions, image, pixels = estimate(args.demo, args.camera, args.plane_z, args.blocks)
    print(json.dumps(positions, indent=2))

    if args.preview:
        preview = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for name, (u, v) in pixels.items():
            cv2.circle(preview, (round(u), round(v)), 9, (0, 255, 255), 2)
            cv2.putText(
                preview,
                name,
                (round(u) + 12, round(v) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), preview):
            raise RuntimeError(f"Could not write preview: {args.preview}")


if __name__ == "__main__":
    main()
