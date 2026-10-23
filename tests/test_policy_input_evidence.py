import ast
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from twingr00t_lab.policy_bridge import CAMERA_MAP
from twingr00t_lab.policy_input_evidence import (
    array_statistics,
    save_policy_input_evidence,
    scene_evidence,
    scene_video_title,
)


def sample_inputs():
    frames = np.arange(36, dtype=np.uint8).reshape(1, 1, 3, 4, 3)
    return {"video": {name: frames.copy() for name in CAMERA_MAP}}, {
        key: frames[0].copy() for key in CAMERA_MAP.values()
    }


def test_scene_diagnostic_cannot_claim_a_verified_source_or_factory_result():
    factory = scene_evidence("factory")
    diagnostic = scene_evidence("no-factory-dressing")
    assert factory["scene_style"] == "factory"
    assert factory["scene_diagnostic_only"] is False
    assert diagnostic["scene_diagnostic_only"] is True
    assert diagnostic["source_scene_equivalence_verified"] is False
    assert "DIAGNOSTIC" in scene_video_title("no-factory-dressing", "train")
    assert "Factory sorting" not in scene_video_title("no-factory-dressing", "train")
    assert "Factory sorting" in scene_video_title("factory", "heldout")
    with pytest.raises(ValueError, match="Unknown scene"):
        scene_evidence("source")


def test_scene_spawner_preserves_light_and_factory_default_without_isaac(tmp_path, monkeypatch):
    """Exercise only the spawner function with a CPU stub for the Isaac API."""
    source = Path(__file__).resolve().parents[1] / "scripts/run_aloha_n17_closedloop.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "spawn_factory_background"
    )
    calls = []

    def spawn_config(**kwargs):
        return SimpleNamespace(func=lambda path, *args, **kw: calls.append((path, kwargs)), **kwargs)

    sim = ModuleType("isaaclab.sim")
    sim.DomeLightCfg = spawn_config
    sim.CuboidCfg = spawn_config
    sim.PreviewSurfaceCfg = spawn_config
    parent = ModuleType("isaaclab")
    parent.sim = sim
    monkeypatch.setitem(sys.modules, "isaaclab", parent)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    hdr = tmp_path / "vendor/RoboTwin/assets/texture/base0.exr"
    hdr.parent.mkdir(parents=True)
    hdr.touch()
    namespace = {"Path": Path, "scene_evidence": scene_evidence, "__file__": str(tmp_path / "scripts/run.py")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    spawn = namespace["spawn_factory_background"]
    spawn("no-factory-dressing")
    assert len(calls) == 1
    assert calls[0][0] == "/World/Light"
    assert calls[0][1] == {"intensity": 3000.0, "color": (0.75, 0.75, 0.75), "texture_file": str(hdr)}
    calls.clear()
    spawn()
    assert calls[0][0] == "/World/Light"
    assert "/World/Factory/Floor" in [path for path, _ in calls]
    assert "/World/Factory/BackWall" in [path for path, _ in calls]
    assert "/World/Factory/StorageRack0" in [path for path, _ in calls]


def test_saved_snapshots_are_exact_model_input_with_ranges_and_hashes(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    policy_input, raw = sample_inputs()
    output = tmp_path / "demo.mp4"
    result = save_policy_input_evidence(policy_input, raw, output, scene_style="factory", policy_call_pending=True)
    assert result["control_step"] == 0
    assert result["policy_call_pending"] is True
    assert json.loads(output.with_suffix(".policy_input.json").read_text()) == result
    for name in CAMERA_MAP:
        camera = result["cameras"][name]
        path = output.with_name(f"demo.{name}.png")
        with Image.open(path) as image:
            np.testing.assert_array_equal(np.asarray(image), policy_input["video"][name][0, 0])
        assert camera["png_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert camera["model_input"] == {
            "dtype": "uint8",
            "shape": [1, 1, 3, 4, 3],
            "min": 0,
            "max": 35,
            "mean": 17.5,
        }
        assert camera["pre_bridge"]["shape"] == [1, 3, 4, 3]
        assert camera["model_input_range_le_one"] is False


def test_dark_conversion_is_observable_not_silently_corrected(tmp_path):
    pytest.importorskip("PIL.Image")
    policy_input, raw = sample_inputs()
    for name, raw_name in CAMERA_MAP.items():
        raw[raw_name] = raw[raw_name].astype(np.float32) / 255
        policy_input["video"][name] = policy_input["video"][name].astype(np.float32) / 255
        policy_input["video"][name] = policy_input["video"][name].astype(np.uint8)
    result = save_policy_input_evidence(
        policy_input, raw, tmp_path / "dark.mp4", scene_style="no-factory-dressing", policy_call_pending=False
    )
    for camera in result["cameras"].values():
        assert camera["pre_bridge"]["dtype"] == "float32"
        assert 0 < camera["pre_bridge"]["max"] < 1
        assert camera["model_input"]["max"] == 0
        assert camera["model_input_range_le_one"] is True
    assert result["policy_call_pending"] is False


def test_invalid_camera_or_existing_evidence_fails_before_writing(tmp_path):
    pytest.importorskip("PIL.Image")
    policy_input, raw = sample_inputs()
    policy_input["video"]["right_wrist"] = policy_input["video"]["right_wrist"].astype(np.float32)
    with pytest.raises(ValueError, match="Expected uint8"):
        save_policy_input_evidence(
            policy_input, raw, tmp_path / "x.mp4", scene_style="factory", policy_call_pending=True
        )
    assert not list(tmp_path.iterdir())
    policy_input, raw = sample_inputs()
    existing = tmp_path / "x.head.png"
    existing.write_bytes(b"existing evidence")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_policy_input_evidence(
            policy_input, raw, tmp_path / "x.mp4", scene_style="factory", policy_call_pending=True
        )
    assert existing.read_bytes() == b"existing evidence"
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize("values", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_invalid_ranges_fail(values):
    with pytest.raises(ValueError, match="non-empty finite"):
        array_statistics(values)
