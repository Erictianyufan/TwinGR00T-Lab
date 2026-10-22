"""Guardrails for the genuine model-inference demo launcher."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_g1_n16_live_demo.py"
SPEC = importlib.util.spec_from_file_location("run_g1_n16_live_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_checkpoint_rejects_old_default_branch(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "gr00t_n1_5"}')
    with pytest.raises(ValueError, match="gn1_6"):
        launcher.checkpoint_manifest(tmp_path)


def test_model_checkpoint_and_commands_have_no_replay_route(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "Gr00tN1d6"}')
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    for name in (
        "model-00001-of-00002.safetensors", "processor_config.json",
        "embodiment_id.json", "statistics.json",
    ):
        (tmp_path / name).write_bytes(b"test")
    manifest = launcher.checkpoint_manifest(tmp_path)
    assert manifest["model_type"] == "Gr00tN1d6"

    args = argparse.Namespace(
        policy_python=Path("/opt/policy/bin/python"),
        sim_python=Path("/opt/sim/bin/python"),
        checkpoint=tmp_path,
        arena_root=Path("/opt/arena"),
        port=5555,
        steps=100,
        seed=10,
        output_dir=tmp_path / "out",
        viewport_video=True,
    )
    server = launcher.server_command(args)
    runner = launcher.runner_command(args)
    assert "--model-path" in server
    assert "--dataset-path" not in server
    assert "--record_camera_video" in runner
    assert "--record_viewport_video" in runner
    assert any("Gr00tRemoteClosedloopPolicy" in part for part in runner)
