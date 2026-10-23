import hashlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from twingr00t_lab.grasp_acceptance import GraspAcceptanceConfig, evaluate_grasp_sorting
from twingr00t_lab.model_identity import build_served_identity, checkpoint_identity


def valid_evidence():
    initial = np.asarray([[0.2, 0.1, 0.765], [-0.2, 0.1, 0.765], [0, 0.1, 0.765]])
    positions = np.tile(initial, (100, 1, 1))
    eef = np.full((100, 2, 3), 10.0)
    states = np.zeros((100, 14))
    states[:, 0] = np.linspace(0, 0.2, 100)
    states[:, 7] = np.linspace(0, 0.2, 100)
    states[:, (6, 13)] = 0.048
    for block, (start, arm) in enumerate(((10, 0), (30, 1), (50, 1))):
        positions[start:, block] = [-0.08 + block * 0.08, 0, 0.765]
        positions[start : start + 10, block, 2] += 0.05
        eef[start : start + 10, arm] = positions[start : start + 10, block] + [0, 0, 0.04]
        states[start : start + 10, (6, 13)[arm]] = 0.02
    delta = positions[:, 1:, :2] - positions[:, :-1, :2]
    sorting = np.all((delta[:, :, 0] > 0) & (delta[:, :, 0] < 0.13) & (abs(delta[:, :, 1]) < 0.03), axis=1)
    return {
        "initial_block_positions": initial,
        "block_positions": positions,
        "block_linear_velocities": np.diff(positions, axis=0, prepend=initial[None]) / 0.02,
        "block_angular_velocities": np.zeros((100, 3, 3)),
        "eef_positions": eef,
        "measured_states": states,
        "sorting_predicate": sorting,
    }


def test_full_bimanual_lift_sort_release_and_settle_passes():
    result = evaluate_grasp_sorting(**valid_evidence(), step_dt=0.02)
    assert result["success"]
    assert result["object_grasped"] == [True, True, True]
    assert result["arm_grasp_evidence"] == [True, True]
    assert result["direct_contact_sensor_verified"] is False
    assert result["video_review_required"] is True
    assert result["stable_window_steps"] == 26


@pytest.mark.parametrize(
    "failure", ["push", "open_gripper", "single_arm", "bounce", "moving", "airborne", "unreleased", "unsorted"]
)
def test_shortcuts_and_unstable_placements_are_not_success(failure):
    evidence = valid_evidence()
    if failure == "push":
        evidence["block_positions"][:, :, 2] = 0.765
    elif failure == "open_gripper":
        evidence["measured_states"][:, (6, 13)] = 0.048
    elif failure == "single_arm":
        evidence["eef_positions"][:, 0] = np.minimum(evidence["eef_positions"][:, 0], evidence["eef_positions"][:, 1])
        evidence["eef_positions"][:, 1] = 10
        evidence["measured_states"][:, 6] = np.min(evidence["measured_states"][:, (6, 13)], axis=1)
    elif failure == "bounce":
        evidence["block_positions"][11:20, 0, 2] = 0.765
    elif failure == "moving":
        evidence["block_linear_velocities"][-1, 0, 0] = 0.1
    elif failure == "airborne":
        evidence["block_positions"][-30:, :, 2] += 0.1
    elif failure == "unreleased":
        evidence["measured_states"][-30:, 6] = 0.02
    else:
        evidence["block_positions"][-30:, :, 0] *= -1
        evidence["sorting_predicate"][-30:] = True  # Even a wrong stored flag cannot override geometry.
    assert not evaluate_grasp_sorting(**evidence, step_dt=0.02)["success"]


def test_nan_measurements_and_nonpositive_time_are_rejected():
    evidence = valid_evidence()
    evidence["eef_positions"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="end effectors"):
        evaluate_grasp_sorting(**evidence, step_dt=0.02)
    with pytest.raises(ValueError, match="step_dt"):
        evaluate_grasp_sorting(**valid_evidence(), step_dt=0)


def test_grasp_milestone_does_not_require_full_sorting():
    evidence = {
        key: value[:25] if key != "initial_block_positions" else value for key, value in valid_evidence().items()
    }
    result = evaluate_grasp_sorting(**evidence, step_dt=0.02)
    assert result["grasp_milestone"]
    assert not result["success"]


@pytest.mark.parametrize("sample_count,passes", [(4, False), (5, True)])
def test_lift_duration_measures_intervals_between_samples(sample_count, passes):
    evidence = valid_evidence()
    for block, start in enumerate((10, 30, 50)):
        evidence["block_positions"][start + sample_count : start + 10, block, 2] = 0.765
    result = evaluate_grasp_sorting(**evidence, step_dt=0.02)
    # Four samples cover only 0.06s; five cover the required 0.08s.
    assert result["required_lift_steps"] == 5
    assert result["success"] is passes


def verifier_module():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_n17_recording.py"
    spec = importlib.util.spec_from_file_location("verify_n17_recording", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recording_fixture(tmp_path, monkeypatch):
    verifier = verifier_module()
    evidence = valid_evidence()
    actions = np.tile(np.linspace(0, 0.2, 100)[:, None], (1, 16)).astype(np.float32)
    chunks = actions.reshape(10, 10, 16)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model_config = {
        "model_type": "Gr00tN1d7",
        "architectures": ["Gr00tN1d7"],
        "model_name": "nvidia/Cosmos-Reason2-2B",
    }
    (checkpoint / "config.json").write_text(json.dumps(model_config))
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        (checkpoint / name).write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"small CPU-only identity fixture")
    model = type("Gr00tN1d7", (), {})()
    model.config = SimpleNamespace(**model_config)
    model.action_head = SimpleNamespace(num_inference_timesteps=8)
    policy = SimpleNamespace(model=model, processor=SimpleNamespace(), embodiment_tag="new_embodiment")
    identity = build_served_identity(policy, checkpoint, flow_steps=8, sample_average=1, seed=42)
    manifest = {
        "recording_schema_version": 2,
        "scene_style": "factory",
        "scene_diagnostic_only": False,
        "status": "success",
        "controller_source": "GR00T_N1.7_model_inference",
        "model_identity": identity,
        "expected_checkpoint_identity": checkpoint_identity(checkpoint),
        "model_sampling": {"flow_steps": 8, "sample_average": 1},
        "policy_seed": 42,
        "replay_used": False,
        "recorded_actions_used_after_reset": False,
        "steps_executed": 100,
        "step_dt": 0.02,
        "stop_reason": "grasp_sort_and_stable_placement",
        "video_review": "passed",
        "record_fps": 25.0,
        "record_every": 2,
        "video_sha256": hashlib.sha256(b"mock video").hexdigest(),
        "video_frame_count": 50,
        "expected_video_frames": 50,
        "expected_video_duration_s": 2.0,
        "acceptance": evaluate_grasp_sorting(**evidence, step_dt=0.02),
        "final_block_positions_m": {f"block{i + 1}": evidence["block_positions"][-1, i].tolist() for i in range(3)},
        "action_safety": {"low_pass_alpha": 1.0, "max_joint_step": 1.0, "max_gripper_step": 1.0},
        "policy_calls": [
            {
                "executed_horizon": 10,
                "action_sha256": hashlib.sha256(chunk.tobytes()).hexdigest(),
                "raw_action_sha256": hashlib.sha256(chunk.tobytes()).hexdigest(),
                "model_identity_id": identity["identity_id"],
            }
            for chunk in chunks
        ],
    }
    np.savez_compressed(
        tmp_path / "video.npz",
        **evidence,
        actions=actions,
        policy_chunks=chunks,
        raw_policy_chunks=chunks,
        clip_lower=np.full(14, -1, dtype=np.float32),
        clip_upper=np.full(14, 1, dtype=np.float32),
        video_control_steps=np.arange(1, 101, 2),
        initial_action=np.zeros(16),
        block_names=np.asarray(["block1", "block2", "block3"]),
    )
    (tmp_path / "video.mp4").write_bytes(b"mock video")
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout='{"format":{"duration":2.0},"streams":[{"nb_read_frames":"50","avg_frame_rate":"25/1"}]}'
        ),
    )
    path = tmp_path / "video.json"
    path.write_text(json.dumps(manifest))
    return verifier, path, manifest


def test_verifier_recomputes_acceptance_and_rejects_reset_or_tampering(recording_fixture):
    verifier, path, manifest = recording_fixture
    assert verifier.verify_recording(path)["grasp_acceptance"]["success"]
    unreviewed = deepcopy(manifest)
    unreviewed["video_review"] = "pending"
    path.write_text(json.dumps(unreviewed))
    with pytest.raises(ValueError, match="video_review=passed"):
        verifier.verify_recording(path)
    diagnostic = verifier.verify_recording(path, allow_unreviewed=True)
    assert diagnostic["status"] == "kinematic_pass_pending_review"
    assert verifier.verify_recording(path, require_success=False)["status"] == "kinematic_pass_pending_review"
    rejected = deepcopy(manifest)
    rejected["video_review"] = "failed"
    path.write_text(json.dumps(rejected))
    with pytest.raises(ValueError, match="video_review=passed"):
        verifier.verify_recording(path, allow_unreviewed=True)
    altered = deepcopy(manifest)
    altered["stop_reason"] = "unexpected_termination"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="termination"):
        verifier.verify_recording(path)
    altered = deepcopy(manifest)
    altered["acceptance"]["max_lift_m"][0] = 99
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="disagrees"):
        verifier.verify_recording(path)


def test_one_millimeter_lift_with_relaxed_manifest_threshold_is_diagnostic_only(recording_fixture):
    verifier, path, manifest = recording_fixture
    trace_path = path.with_suffix(".npz")
    with np.load(trace_path) as trace:
        arrays = dict(trace)
    for block, start in enumerate((10, 30, 50)):
        arrays["block_positions"][start : start + 10, block, 2] = 0.766
        arm = 0 if block == 0 else 1
        arrays["eef_positions"][start : start + 10, arm] = arrays["block_positions"][start : start + 10, block] + [
            0,
            0,
            0.04,
        ]
    arrays["block_linear_velocities"] = (
        np.diff(arrays["block_positions"], axis=0, prepend=arrays["initial_block_positions"][None]) / 0.02
    )
    evidence = {key: arrays[key] for key in valid_evidence()}
    assert evaluate_grasp_sorting(**evidence, step_dt=0.02)["success"] is False
    weak_report = evaluate_grasp_sorting(**evidence, step_dt=0.02, config=GraspAcceptanceConfig(min_lift_m=0.0005))
    assert weak_report["success"] is True
    np.testing.assert_allclose(weak_report["max_lift_m"], [0.001] * 3)
    manifest["acceptance"] = weak_report
    np.savez_compressed(trace_path, **arrays)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="official default GraspAcceptanceConfig"):
        verifier.verify_recording(path)
    result = verifier.verify_recording(path, require_success=False)
    assert result["grasp_acceptance"]["success"] is True
    assert result["status"] == "kinematic_pass_diagnostic_only"
    assert result["formal_acceptance"] is False


@pytest.mark.parametrize("config", [None, {}, {"min_lift_m": 0.025}])
def test_formal_acceptance_rejects_missing_or_partial_default_config(recording_fixture, config):
    verifier, path, manifest = recording_fixture
    if config is None:
        manifest["acceptance"].pop("config")
    else:
        manifest["acceptance"]["config"] = config
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="official default GraspAcceptanceConfig"):
        verifier.verify_recording(path)


def test_formal_identity_requires_service_checkpoint_and_every_response(recording_fixture):
    verifier, path, manifest = recording_fixture
    result = verifier.verify_recording(path)
    assert result["model_identity_verified"] is True
    assert result["model_identity_id"] == manifest["model_identity"]["identity_id"]
    assert result["checkpoint_id"] == manifest["expected_checkpoint_identity"]["checkpoint_id"]
    manifest.pop("model_identity")
    manifest.pop("expected_checkpoint_identity")
    for call in manifest["policy_calls"]:
        call.pop("model_identity_id")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Formal acceptance requires verified model identity"):
        verifier.verify_recording(path)
    result = verifier.verify_recording(path, require_success=False)
    assert result["model_identity_verified"] is False
    assert result["model_identity_status"] == "legacy_identity_missing_unverified"
    assert result["model_identity_id"] is None
    assert result["formal_acceptance"] is False


@pytest.mark.parametrize(
    "missing",
    ["model_identity", "expected_checkpoint_identity", "model_sampling", "flow_steps", "sample_average", "policy_seed"],
)
def test_partial_identity_evidence_cannot_downgrade_to_legacy(recording_fixture, missing):
    verifier, path, manifest = recording_fixture
    if missing in ("flow_steps", "sample_average"):
        manifest["model_sampling"].pop(missing)
    else:
        manifest.pop(missing)
    path.write_text(json.dumps(manifest))
    for require_success in (False, True):
        with pytest.raises(ValueError):
            verifier.verify_recording(path, require_success=require_success)


@pytest.mark.parametrize("alteration", ["flow_steps", "sample_average", "policy_seed", "call_id", "missing_call_id"])
def test_identity_sampling_and_individual_response_mismatches_fail(recording_fixture, alteration):
    verifier, path, manifest = recording_fixture
    if alteration in ("flow_steps", "sample_average"):
        manifest["model_sampling"][alteration] += 1
    elif alteration == "policy_seed":
        manifest["policy_seed"] += 1
    elif alteration == "call_id":
        manifest["policy_calls"][5]["model_identity_id"] = "0" * 64
    else:
        manifest["policy_calls"][5].pop("model_identity_id")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verifier.verify_recording(path)


def test_server_identity_cannot_be_reused_for_another_checkpoint(recording_fixture):
    verifier, path, manifest = recording_fixture
    checkpoint = path.parent / "checkpoint"
    (checkpoint / "model.safetensors").write_bytes(b"different but validly hashed checkpoint fixture")
    manifest["expected_checkpoint_identity"] = checkpoint_identity(checkpoint)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from local expected checkpoint"):
        verifier.verify_recording(path)


def test_tampered_model_identity_digest_is_rejected(recording_fixture):
    verifier, path, manifest = recording_fixture
    manifest["model_identity"]["loaded_model"]["model_type"] = "Gr00tN1d6"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity digest mismatch"):
        verifier.verify_recording(path)


@pytest.mark.parametrize(
    "scene,diagnostic",
    [("no-factory-dressing", True), ("factory", True), (None, None), ("factory", None), (None, False)],
)
def test_nonformal_or_unrecorded_scene_never_passes_formal_factory_acceptance(recording_fixture, scene, diagnostic):
    verifier, path, manifest = recording_fixture
    if scene is None:
        manifest.pop("scene_style")
    else:
        manifest["scene_style"] = scene
    if diagnostic is None:
        manifest.pop("scene_diagnostic_only")
    else:
        manifest["scene_diagnostic_only"] = diagnostic
    path.write_text(json.dumps(manifest))
    # Even an otherwise complete, visually reviewed kinematic pass is not a
    # formal factory demo. --allow-unreviewed cannot bypass the scene gate.
    with pytest.raises(ValueError, match="scene_style=factory"):
        verifier.verify_recording(path, allow_unreviewed=True)
    result = verifier.verify_recording(path, require_success=False)
    assert result["status"] == "kinematic_pass_diagnostic_scene"
    assert result["formal_acceptance"] is False
    assert result["formal_factory_scene"] is False


def test_allow_failed_is_diagnostic_even_for_a_valid_factory_success(recording_fixture):
    verifier, path, _ = recording_fixture
    assert verifier.verify_recording(path)["formal_acceptance"] is True
    result = verifier.verify_recording(path, require_success=False)
    assert result["formal_factory_scene"] is True
    assert result["status"] == "kinematic_pass_diagnostic_only"
    assert result["formal_acceptance"] is False


@pytest.mark.parametrize("diagnostic", [0, 1, "false", "true"])
def test_scene_diagnostic_flag_requires_a_real_boolean(recording_fixture, diagnostic):
    verifier, path, manifest = recording_fixture
    manifest["scene_diagnostic_only"] = diagnostic
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="JSON boolean"):
        verifier.verify_recording(path, require_success=False)


def test_policy_input_png_hashes_use_portable_adjacent_files(recording_fixture):
    verifier, path, manifest = recording_fixture
    cameras = {}
    for name in ("head", "left_wrist", "right_wrist"):
        png = path.with_name(f"{path.stem}.{name}.png")
        payload = f"mock {name} PNG".encode()
        png.write_bytes(payload)
        cameras[name] = {"png": f"/remote/output/{png.name}", "png_sha256": hashlib.sha256(payload).hexdigest()}
    manifest["first_policy_input"] = {"cameras": cameras}
    path.write_text(json.dumps(manifest))
    assert verifier.verify_recording(path)["first_policy_input_pngs_verified"] is True
    path.with_name(f"{path.stem}.head.png").write_bytes(b"another image")
    with pytest.raises(ValueError, match="PNG SHA-256 mismatch"):
        verifier.verify_recording(path)


def test_verifier_rejects_rehashed_raw_outputs_that_do_not_explain_clipped_actions(recording_fixture):
    verifier, path, manifest = recording_fixture
    trace_path = path.with_suffix(".npz")
    with np.load(trace_path) as trace:
        arrays = dict(trace)
    arrays["raw_policy_chunks"][0, 0, 0] += 0.2
    manifest["policy_calls"][0]["raw_action_sha256"] = hashlib.sha256(
        arrays["raw_policy_chunks"][0].tobytes()
    ).hexdigest()
    np.savez_compressed(trace_path, **arrays)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not the saved raw"):
        verifier.verify_recording(path)


def test_verifier_rejects_mismatched_mp4_identity(recording_fixture):
    verifier, path, _ = recording_fixture
    path.with_suffix(".mp4").write_bytes(b"a different recording")
    with pytest.raises(ValueError, match="Video SHA-256"):
        verifier.verify_recording(path)


@pytest.mark.parametrize(
    "frames,fps,duration,message",
    [(49, "25/1", 1.96, "frame count"), (50, "50/1", 1.0, "frame rate"), (50, "25/1", 1.0, "duration")],
)
def test_verifier_detects_incomplete_video_despite_matching_identity(
    recording_fixture, monkeypatch, frames, fps, duration, message
):
    verifier, path, _ = recording_fixture
    payload = {"format": {"duration": duration}, "streams": [{"nb_read_frames": frames, "avg_frame_rate": fps}]}
    monkeypatch.setattr(verifier.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(payload)))
    with pytest.raises(ValueError, match=message):
        verifier.verify_recording(path)


def test_video_control_steps_cannot_skip_part_of_the_episode(recording_fixture):
    verifier, path, _ = recording_fixture
    trace_path = path.with_suffix(".npz")
    with np.load(trace_path) as trace:
        arrays = dict(trace)
    arrays["video_control_steps"][20] += 1
    np.savez_compressed(trace_path, **arrays)
    with pytest.raises(ValueError, match="control-step"):
        verifier.verify_recording(path)


def test_legacy_recording_remains_diagnostic_only(recording_fixture):
    verifier, path, manifest = recording_fixture
    manifest.pop("recording_schema_version")
    # Older schemas computed four samples as an 0.08s lift; re-evaluate them
    # under the new rule, but do not discard diagnostic access to old traces.
    manifest["acceptance"]["required_lift_steps"] = 4
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recording_schema_version=2"):
        verifier.verify_recording(path)
    result = verifier.verify_recording(path, require_success=False)
    assert result["status"] == "legacy_diagnostic_unverified"
    assert result["acceptance_report_matches"] is False
    assert result["raw_to_clipped_actions_verified"] is False
    assert result["video_identity_and_completeness_verified"] is False


def test_acceptance_config_is_serializable():
    assert GraspAcceptanceConfig(**json.loads(json.dumps(asdict(GraspAcceptanceConfig())))) == GraspAcceptanceConfig()
