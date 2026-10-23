"""Exhaustive CPU review preparation is never a success/review approval."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image", reason="Visual review tests require the optional review extra")


@pytest.fixture
def review(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_n17_visual_review.py"
    spec = importlib.util.spec_from_file_location("n17_visual_review_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "THUMBNAIL", (16, 12))
    return module


@pytest.fixture
def recording(tmp_path):
    video = tmp_path / "failed_recording.mp4"
    video.write_bytes(b"Only used by mocked decoder, never claimed to be a real MP4")
    manifest = {
        "recording_schema_version": 2,
        "steps_executed": 25,
        "record_every": 2,
        "video_frame_count": 13,
        "expected_video_frames": 13,
        "record_fps": 25.0,
        "step_dt": 0.02,
        "expected_video_duration_s": 13 / 25,
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "video_review": "pending",
        "acceptance": {"success": False},
    }
    path = video.with_suffix(".json")
    path.write_text(json.dumps(manifest))
    return path, tmp_path / "review", manifest


def mock_reader(*, count=13, size=(16, 12), fps=25, duration=13 / 25, closed=None):
    def read(path, *, pix_fmt, output_params):
        assert Path(path).suffix == ".mp4"
        assert pix_fmt == "rgb24" and output_params == ["-vsync", "0"]
        try:
            yield {"size": size, "fps": fps, "duration": duration}
            for index in range(count):
                yield Image.new("RGB", size, (index + 1, 50, 90)).tobytes()
        finally:
            if closed is not None:
                closed.append(True)

    return read


def test_every_decoded_frame_appears_once_in_order_without_modifying_sources(review, recording):
    manifest, output, original = recording
    before = manifest.read_bytes(), manifest.with_suffix(".mp4").read_bytes()
    index = review.prepare_review(manifest, output, reader=mock_reader())
    assert index["status"] == "awaiting_visual_review"
    assert index["source_acceptance_success"] is False and index["source_video_review"] == "pending"
    assert index["source_video_sha256"] == original["video_sha256"]
    assert index["decoded_frame_count"] == 13
    assert [(s["file"], s["first_frame"], s["last_frame"]) for s in index["sheets"]] == [
        ("sheet_0001.png", 1, 12),
        ("sheet_0002.png", 13, 13),
    ]
    entries = [frame for sheet in index["sheets"] for frame in sheet["frames"]]
    assert [frame["frame"] for frame in entries] == list(range(1, 14))
    assert [frame["control_step"] for frame in entries] == list(range(1, 26, 2))
    assert entries[0]["video_time_s"] == 0 and entries[0]["simulation_time_s"] == 0.02
    assert entries[-1]["video_time_s"] == 12 / 25 and entries[-1]["simulation_time_s"] == 25 * 0.02
    for sheet in index["sheets"]:
        with Image.open(output / sheet["file"]) as image:
            for frame in sheet["frames"]:
                x, y = frame["column"] * 16, frame["row"] * (12 + review.CAPTION_HEIGHT)
                # Captions do not obscure the source image, including its bottom row.
                assert image.getpixel((x + 8, y + 11)) == (frame["frame"], 50, 90)
    assert json.loads((output / "index.json").read_text()) == index
    assert before == (manifest.read_bytes(), manifest.with_suffix(".mp4").read_bytes())


@pytest.mark.parametrize("count", [0, 12, 14])
def test_missing_or_extra_frames_leave_no_completed_review_index(review, recording, count):
    manifest, output, _ = recording
    closed = []
    with pytest.raises(ValueError, match="[Dd]ecoded"):
        review.prepare_review(manifest, output, reader=mock_reader(count=count, closed=closed))
    assert not (output / "index.json").exists()
    assert closed == [True]
    assert json.loads(manifest.read_text())["video_review"] == "pending"


@pytest.mark.parametrize(
    "field,value",
    [
        ("recording_schema_version", 1),
        ("video_sha256", "0" * 64),
        ("video_frame_count", 12),
        ("expected_video_frames", 14),
        ("record_every", 3),
        ("record_fps", 50),
        ("step_dt", float("nan")),
        ("expected_video_duration_s", 42),
    ],
)
def test_wrong_source_identity_or_manifest_cadence_is_rejected(review, recording, field, value):
    manifest, output, data = recording
    data[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        review.prepare_review(manifest, output, reader=mock_reader())
    assert not output.exists()


@pytest.mark.parametrize("changes", [{"fps": 50}, {"duration": 8}, {"duration": 0}, {"duration": float("nan")}])
def test_decoded_timing_must_match_manifest(review, recording, changes):
    manifest, output, _ = recording
    with pytest.raises(ValueError):
        review.prepare_review(manifest, output, reader=mock_reader(**changes))
    assert not output.exists()


def test_existing_review_directory_is_never_overwritten(review, recording):
    manifest, output, _ = recording
    output.mkdir()
    sentinel = output / "user_notes.txt"
    sentinel.write_text("Keep this")
    with pytest.raises(FileExistsError):
        review.prepare_review(manifest, output, reader=mock_reader())
    assert sentinel.read_text() == "Keep this"


def test_main_crop_requires_known_recorder_layout(review, recording):
    manifest, output, _ = recording
    with pytest.raises(ValueError, match="1920x1080"):
        review.prepare_review(manifest, output, view="main", reader=mock_reader())
    assert not output.exists()


def test_main_crop_uses_explicit_overview_rectangle_and_records_exclusion(review, recording):
    manifest, output, data = recording
    data.update(steps_executed=1, video_frame_count=1, expected_video_frames=1, expected_video_duration_s=1 / 25)
    manifest.write_text(json.dumps(data))

    def reader(*args, **kwargs):
        yield {"size": (1920, 1080), "fps": 25, "duration": 1 / 25}
        frame = Image.new("RGB", (1920, 1080), "blue")
        frame.paste("red", review.MAIN_CROP)
        yield frame.tobytes()

    result = review.prepare_review(manifest, output, view="main", reader=reader)
    assert result["crop_box_xyxy"] == [0, 60, 1280, 1020]
    assert "excludes the other camera panels" in result["review_limitations"]
    with Image.open(output / "sheet_0001.png") as image:
        assert image.getpixel((8, 6)) == (255, 0, 0)
