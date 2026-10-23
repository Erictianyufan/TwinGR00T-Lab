"""Exercise the post-training CPU guards without starting Torch, SSH, or a GPU."""

import json
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_n17_posttrain_checks.sh"


@pytest.fixture
def guards():
    source = SCRIPT.read_text(encoding="utf-8").split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    namespace = {"__name__": "posttrain_guard_test"}
    exec(compile(source, str(SCRIPT), "exec"), namespace)
    return namespace


@pytest.fixture
def completed_checkpoint(tmp_path):
    output = tmp_path / "training"
    checkpoint = output / "checkpoint-4000"
    checkpoint.mkdir(parents=True)
    processor = output / "processor"
    processor.mkdir()
    visual = {
        "color_jitter_params": dict.fromkeys(("brightness", "contrast", "saturation", "hue"), 0.0),
        "random_rotation_angle": 0,
        "crop_fraction": 0.95,
        "image_crop_size": [230, 230],
        "image_target_size": [256, 256],
        "use_albumentations": True,
    }
    for directory in (checkpoint, processor):
        (directory / "processor_config.json").write_text(json.dumps({"processor_kwargs": visual}))
        for name in ("statistics.json", "embodiment_id.json"):
            (directory / name).write_text(json.dumps({"new_embodiment": 1}))
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 4000}))
    (output / "visual_augmentation_audit.json").write_text(
        json.dumps(
            {
                "status": "verified_color_off",
                "mode": "color-off",
                "effective_processor_source": str(processor / "processor_config.json"),
                "effective_visual_config": visual,
                "crop_and_rotation": "inherited_not_overridden",
            }
        )
    )
    header = json.dumps({"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (checkpoint / "model-00001.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "model-00001.safetensors"}})
    )
    return checkpoint, output


def test_complete_checkpoint_matches_actual_launcher_audit_schema(guards, completed_checkpoint):
    checkpoint, output = completed_checkpoint
    guards["verify_checkpoint"](checkpoint, output, 4000, 0)


@pytest.mark.parametrize("kind", ["truncated", "missing_tensor", "old_audit", "wrong_processor", "stale_stats"])
def test_stale_or_incomplete_checkpoint_never_starts_inference(guards, completed_checkpoint, kind):
    checkpoint, output = completed_checkpoint
    started = 0
    if kind == "truncated":
        shard = checkpoint / "model-00001.safetensors"
        shard.write_bytes(shard.read_bytes()[:-1])  # Nonempty, but one tensor byte is missing.
    elif kind == "missing_tensor":
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"missing_tensor": "model-00001.safetensors"}})
        )
    elif kind == "old_audit":
        started = (output / "visual_augmentation_audit.json").stat().st_mtime + 10
    elif kind == "wrong_processor":
        path = output / "processor/processor_config.json"
        payload = json.loads(path.read_text())
        payload["processor_kwargs"]["color_jitter_params"]["hue"] = 0.08
        path.write_text(json.dumps(payload))
    else:
        (checkpoint / "statistics.json").write_text(json.dumps({"new_embodiment": 2}))
    with pytest.raises(RuntimeError):
        guards["verify_checkpoint"](checkpoint, output, 4000, started)


def test_pid_is_bound_to_training_launcher_and_expected_output(guards, tmp_path):
    process = {
        "state": "S",
        "cwd": tmp_path,
        "argv": ["python", "scripts/launch_aloha_n17_lowmem.py", "--output-dir", "training"],
    }
    guards["verify_training_identity"](process, tmp_path / "training")
    with pytest.raises(RuntimeError, match="output directory"):
        guards["verify_training_identity"](process, tmp_path / "another_run")
    process["argv"][1] = "scripts/serve_aloha_n17.py"
    with pytest.raises(RuntimeError, match="not the N1.7 training launcher"):
        guards["verify_training_identity"](process, tmp_path / "training")
    with pytest.raises(RuntimeError, match="not running"):
        guards["verify_training_identity"](None, tmp_path / "training")


def test_pid_reuse_is_rejected_before_any_checkpoint_read(guards, tmp_path, monkeypatch):
    initial = {
        "state": "S",
        "start_ticks": 123,
        "cwd": tmp_path,
        "argv": ["python", "launch_aloha_n17_lowmem.py", "--output-dir=training"],
    }
    processes = iter((initial, {**initial, "start_ticks": 456}))
    guards["read_process"] = lambda pid: next(processes)
    monkeypatch.setattr(guards["Path"], "read_text", lambda path: "btime 1000\n")
    monkeypatch.setattr(guards["os"], "sysconf", lambda key: 100, raising=False)
    with pytest.raises(RuntimeError, match="PID was reused"):
        guards["wait_for_training"](123, tmp_path / "training")
