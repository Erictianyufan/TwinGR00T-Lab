"""Exercise the post-training CPU guards without starting Torch, SSH, or a GPU."""

import json
import os
import shutil
import struct
import subprocess
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


@pytest.fixture
def diagnostic_guards():
    source = SCRIPT.read_text(encoding="utf-8").split("<<'PY'\n")[2].split("\nPY\n", 1)[0]
    namespace = {"__name__": "posttrain_diagnostic_test"}
    exec(compile(source, str(SCRIPT), "exec"), namespace)
    return namespace


@pytest.fixture
def diagnostic_inputs(tmp_path):
    demos = []
    for split, instruction in (("train", "Use left, right, left arms"), ("heldout", "Use right, left, left arms")):
        root = tmp_path / split
        (root / "data").mkdir(parents=True)
        (root / "instructions").mkdir()
        demo = root / "data/demo_0.hdf5"
        demo.write_bytes(b"input-path-only fixture")
        (root / "instructions/demo_0.json").write_text(json.dumps({"seen": [instruction]}))
        demos.append(demo)
    stats = tmp_path / "factory-aligned/meta/stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}")
    return (*demos, stats)


def test_factory_rgb_inputs_keep_split_specific_prompts_and_actual_stats(diagnostic_guards, diagnostic_inputs):
    result = diagnostic_guards["resolve_diagnostic_config"](*diagnostic_inputs, "0")
    assert result["correct_legacy_jpeg_bgr"] is False
    assert result["observation_action_offset"] == 1
    assert result["offline_prompts"]["train"]["text"] == "Use left, right, left arms"
    assert result["offline_prompts"]["heldout"]["text"] == "Use right, left, left arms"
    assert result["online_prompt"] == result["offline_prompts"]["train"]
    assert result["action_stats"] == str(diagnostic_inputs[2].resolve())
    assert result["initial_state_split"] == "train"
    assert result["status"] == "diagnostic_configuration_not_inference_success"


def test_online_override_is_explicit_and_does_not_change_offline_prompts(diagnostic_guards, diagnostic_inputs):
    result = diagnostic_guards["resolve_diagnostic_config"](*diagnostic_inputs, "1", "  Different online request  ")
    assert result["correct_legacy_jpeg_bgr"] is True
    assert result["online_prompt"] == {"text": "Different online request", "source": "POLICY_PROMPT_override"}
    assert result["offline_prompts"]["heldout"]["text"] == "Use right, left, left arms"


@pytest.mark.parametrize("value", ["", "true", "false", "2", "01"])
def test_ambiguous_color_switch_is_rejected(diagnostic_guards, diagnostic_inputs, value):
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        diagnostic_guards["resolve_diagnostic_config"](*diagnostic_inputs, value)


def test_missing_factory_instruction_never_falls_back_to_old_fixed_prompt(diagnostic_guards, diagnostic_inputs):
    train, _, _ = diagnostic_inputs
    instruction = train.parent.parent / "instructions/demo_0.json"
    instruction.write_text(json.dumps({"seen": []}))
    with pytest.raises(ValueError, match=r"seen\[0\]"):
        diagnostic_guards["resolve_diagnostic_config"](*diagnostic_inputs, "0")


@pytest.mark.parametrize("factory", [False, True])
def test_shell_color_array_and_path_overrides_match_dataset_contract(factory):
    git_bash = Path("D:/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable for CPU-only shell parameter test")
    text = SCRIPT.read_text(encoding="utf-8")
    block = 'train_demo="${TRAIN_DEMO:' + text.split('train_demo="${TRAIN_DEMO:', 1)[1].split('[[ "${gpu_id}"', 1)[0]
    env = os.environ.copy()
    keys = ("TRAIN_DEMO", "HELDOUT_DEMO", "ACTION_STATS", "CORRECT_LEGACY_JPEG_BGR")
    for key in keys:
        env.pop(key, None)
    if factory:
        env.update(
            TRAIN_DEMO="/factory/data/demo_0.hdf5",
            HELDOUT_DEMO="/heldout/data/demo_9.hdf5",
            ACTION_STATS="/factory-aligned/meta/stats.json",
            CORRECT_LEGACY_JPEG_BGR="0",
        )
    command = (
        "project_root=/project\n"
        + block
        + """
printf '%s\n' "${train_demo}" "${heldout_demo}" "${action_stats}"
printf '%s\n' "${correct_legacy_jpeg_bgr}" "${#legacy_color_args[@]}" "${legacy_color_args[*]}"
"""
    )
    result = subprocess.run([bash, "-c", command], capture_output=True, text=True, check=True, env=env)
    lines = result.stdout.splitlines()
    if factory:
        assert lines[:5] == [env["TRAIN_DEMO"], env["HELDOUT_DEMO"], env["ACTION_STATS"], "0", "0"]
        assert len(lines) == 6 and lines[5] == ""
    else:
        assert lines[0].endswith("blocks_ranking_rgb-vision-batch/data/demo_0.hdf5")
        assert lines[1].endswith("blocks_ranking_rgb-vision-ep1/data/demo_0.hdf5")
        assert lines[2].endswith("blocks_ranking_rgb-aligned-v1/meta/stats.json")
        assert lines[3:] == ["1", "1", "--correct-legacy-jpeg-bgr"]


def test_diagnostics_preserve_scene_seed_identity_and_never_publish():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"${legacy_color_args[@]}" --observation-action-offset 1' in text
    assert 'ACTION_STATS="${action_stats}"' in text
    assert "for style in factory no-factory-dressing" in text
    assert "POLICY_SEED=42 SIM_SEED=42" in text
    assert 'INITIAL_STATE_SPLIT=train POLICY_PROMPT="${prompt}"' in text
    assert "--allow-failed" in text and "set -o noclobber" in text
    assert "git push" not in text and "gh release" not in text
