"""Test display-only composition without importing Torch or the Isaac runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
ImageFont = pytest.importorskip("PIL.ImageFont")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_live_multiview.py"


@pytest.fixture
def display():
    functions = {"load_font", "to_rgb", "letterbox", "draw_label", "fit_text_line", "compose_frame"}
    constants = {"CANVAS_SIZE", "HEADER_HEIGHT", "MAIN_SIZE", "SIDE_SIZE", "CAMERA_KEYS", "TASK_LABELS"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        or (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
        )
    ]
    namespace = {
        "Path": Path, "np": np, "Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont,
        "torch": SimpleNamespace(Tensor=type("UnusedTensor", (), {})),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def marker_frame():
    frame = np.full((480, 640, 3), (72, 92, 112), dtype=np.uint8)
    frame[:24, :24] = (255, 0, 0)
    frame[:24, -24:] = (0, 255, 0)
    frame[-24:, :24] = (0, 0, 255)
    frame[-24:, -24:] = (255, 255, 0)
    return frame


def observations(display):
    return {key: marker_frame()[None] for key in display["CAMERA_KEYS"]}


def test_five_camera_panels_fit_content_area_and_footer_is_separate(display, monkeypatch):
    calls = []
    original_paste = Image.Image.paste

    def capture_paste(self, image, box=None, mask=None):
        if self.size == display["CANVAS_SIZE"]:
            calls.append((image.size, box))
        return original_paste(self, image, box, mask)

    monkeypatch.setattr(Image.Image, "paste", capture_paste)
    result = display["compose_frame"](observations(display), 0, 100, 31.25, "blocks_ranking_rgb", "Sort blocks")
    assert result.mode == "RGB"
    assert result.size == (1920, 1080)
    expected = [
        ((1280, 960), (0, 60)),
        ((640, 320), (1280, 60)),
        ((640, 320), (1280, 380)),
        ((320, 320), (1280, 700)),
        ((320, 320), (1600, 700)),
    ]
    assert calls[:5] == expected
    for (width, height), (x, y) in calls[:5]:
        assert 0 <= x < x + width <= 1920
        assert 60 <= y < y + height <= 1020
    assert calls[5:] == [((1920, 60), (0, 0)), ((1920, 60), (0, 1020))]


@pytest.mark.parametrize(
    ("panel_size", "position"),
    [((1280, 960), (0, 60)), ((640, 320), (1280, 60)), ((640, 320), (1280, 380)),
     ((320, 320), (1280, 700)), ((320, 320), (1600, 700))],
)
def test_each_camera_preserves_all_four_corners_and_bottom_edge(display, panel_size, position):
    result = display["compose_frame"](observations(display), 0, 100, 31.25, "blocks_ranking_rgb", "Sort blocks")
    scale = min(panel_size[0] / 640, panel_size[1] / 480)
    width, height = round(640 * scale), round(480 * scale)
    x = position[0] + (panel_size[0] - width) // 2
    y = position[1] + (panel_size[1] - height) // 2
    assert result.getpixel((x, y)) == (255, 0, 0)
    assert result.getpixel((x + width - 1, y)) == (0, 255, 0)
    assert result.getpixel((x, y + height - 1)) == (0, 0, 255)
    assert result.getpixel((x + width - 1, y + height - 1)) == (255, 255, 0)
    assert result.getpixel((x + width // 2, y + height - 1)) == (72, 92, 112)


def test_long_multiline_text_cannot_cover_camera_content_or_change_inputs(display):
    inputs = observations(display)
    copies = {key: value.copy() for key, value in inputs.items()}
    compose = display["compose_frame"]
    ordinary = compose(inputs, 7, 100, 31.25, "blocks_ranking_rgb", "Sort blocks")
    long = compose(
        inputs, 7, 100, 31.25, "blocks_ranking_rgb", "Long instruction\n" * 500,
        title_override="SYNTHETIC LAYOUT TEST ONLY " * 200,
    )
    np.testing.assert_array_equal(np.asarray(ordinary)[60:1020], np.asarray(long)[60:1020])
    assert not np.array_equal(np.asarray(ordinary)[:60], np.asarray(long)[:60])
    assert not np.array_equal(np.asarray(ordinary)[1020:], np.asarray(long)[1020:])
    for key in inputs:
        np.testing.assert_array_equal(inputs[key], copies[key])


def test_text_width_is_bounded_and_short_text_is_unchanged(display):
    draw = ImageDraw.Draw(Image.new("RGB", (1920, 60)))
    font = display["load_font"](18)
    fit = display["fit_text_line"]
    assert fit(draw, "Short instruction", font, 1876) == "Short instruction"
    assert fit(draw, "One\n two\tthree", font, 1876) == "One two three"
    shortened = fit(draw, "Very long instruction " * 500, font, 1876)
    assert shortened.endswith("...")
    left, _, right, _ = draw.textbbox((0, 0), shortened, font=font)
    assert right - left <= 1876
    assert fit(draw, "Does not fit", font, 0) == ""


def test_legacy_six_argument_replay_call_and_online_override_remain_supported(display):
    compose = display["compose_frame"]
    args = (observations(display), 0, 10, 31.25, "unknown_task", "Place object")
    assert compose(*args).size == (1920, 1080)
    assert compose(*args, title_override="SYNTHETIC - NOT INFERENCE").size == (1920, 1080)
