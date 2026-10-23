"""Content and actual-model binding can be tested without importing Torch."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from twingr00t_lab.model_identity import (
    bind_action_identity,
    build_served_identity,
    checkpoint_identity,
    validate_model_identity,
)


def make_checkpoint(root: Path, *, processor_subdirectory: bool = False, sharded: bool = False) -> Path:
    root.mkdir(parents=True)
    config = {
        "model_type": "Gr00tN1d7",
        "architectures": ["Gr00tN1d7"],
        "model_name": "nvidia/Cosmos-Reason2-2B",
        "token": "must-not-appear-in-identity",
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    processor_dir = root / "processor" if processor_subdirectory else root
    processor_dir.mkdir(exist_ok=True)
    (processor_dir / "processor_config.json").write_text('{"use_relative_action": true}', encoding="utf-8")
    (processor_dir / "statistics.json").write_text('{"action": {"mean": [0.1]}}', encoding="utf-8")
    (processor_dir / "embodiment_id.json").write_text('{"new_embodiment": 32}', encoding="utf-8")
    if sharded:
        shards = {"layer.a": "model-00001-of-00002.safetensors", "layer.b": "model-00002-of-00002.safetensors"}
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": shards}), encoding="utf-8")
        for index, name in enumerate(shards.values()):
            (root / name).write_bytes(f"test placeholder shard {index}".encode())
    else:
        (root / "model.safetensors").write_bytes(b"test placeholder: hash contents, not a GPU tensor parser")
    return root


def make_policy(*, flow_steps: int = 8) -> SimpleNamespace:
    # The production function checks the actual class, not just configuration text.
    model = type("Gr00tN1d7", (), {})()
    model.config = SimpleNamespace(
        model_type="Gr00tN1d7", architectures=["Gr00tN1d7"], model_name="nvidia/Cosmos-Reason2-2B",
    )
    model.action_head = SimpleNamespace(num_inference_timesteps=flow_steps)
    return SimpleNamespace(model=model, processor=SimpleNamespace(), embodiment_tag="new_embodiment")


def make_identity(root: Path) -> dict:
    return build_served_identity(make_policy(), root, flow_steps=8, sample_average=1, seed=42)


def test_identity_is_stable_content_bound_and_does_not_expose_config_or_paths(tmp_path: Path) -> None:
    root = make_checkpoint(tmp_path / "first")
    copy_root = make_checkpoint(tmp_path / "different-location")
    identity = make_identity(root)
    assert identity == make_identity(copy_root)
    validate_model_identity(
        identity, expected_checkpoint_identity=checkpoint_identity(copy_root),
        expected_flow_steps=8, expected_sample_average=1, expected_seed=42,
    )
    serialized = json.dumps(identity)
    assert "must-not-appear" not in serialized
    assert "token" not in serialized
    assert str(root) not in serialized
    assert identity["loaded_model"]["model_class"].endswith(".Gr00tN1d7")


def test_effective_processor_root_precedence_and_fallback(tmp_path: Path) -> None:
    root = make_checkpoint(tmp_path / "checkpoint", processor_subdirectory=True)
    first = checkpoint_identity(root)
    assert first["processor_config_path"] == "processor/processor_config.json"
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        (root / name).write_bytes((root / "processor" / name).read_bytes())
    second = checkpoint_identity(root)
    assert second["processor_config_path"] == "processor_config.json"
    assert "processor/statistics.json" not in second["files"]
    assert second["checkpoint_id"] != first["checkpoint_id"]


def test_hashes_all_shards_and_processor_statistics(tmp_path: Path) -> None:
    root = make_checkpoint(tmp_path / "checkpoint", sharded=True)
    before = checkpoint_identity(root)
    assert len(before["files"]) == 7
    shard = root / "model-00002-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes() + b"changed")
    after = checkpoint_identity(root)
    assert before["checkpoint_id"] != after["checkpoint_id"]
    with pytest.raises(ValueError, match="local expected checkpoint"):
        validate_model_identity(make_identity(root), expected_checkpoint_identity=before)
    (root / "statistics.json").write_text('{"action": {"mean": [0.2]}}', encoding="utf-8")
    assert checkpoint_identity(root)["checkpoint_id"] != after["checkpoint_id"]


@pytest.mark.parametrize("filename", ["model.safetensors", "processor_config.json", "statistics.json"])
def test_missing_or_incomplete_files_rejected(tmp_path: Path, filename: str) -> None:
    root = make_checkpoint(tmp_path / "checkpoint")
    (root / filename).write_bytes(b"")
    with pytest.raises(ValueError):
        checkpoint_identity(root)
    (root / filename).unlink()
    with pytest.raises(ValueError):
        checkpoint_identity(root)


@pytest.mark.parametrize("bad_path", ["../escape.safetensors", "/absolute.safetensors", "C:/model.safetensors"])
def test_index_cannot_hash_outside_checkpoint(tmp_path: Path, bad_path: str) -> None:
    root = make_checkpoint(tmp_path / "checkpoint", sharded=True)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": bad_path}}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="safe relative"):
        checkpoint_identity(root)


def test_rejects_actual_class_mismatch_even_when_config_claims_n17(tmp_path: Path) -> None:
    root = make_checkpoint(tmp_path / "checkpoint")
    policy = make_policy()
    policy.model = SimpleNamespace(config=policy.model.config, action_head=policy.model.action_head)
    with pytest.raises(ValueError, match="Actual loaded model class"):
        build_served_identity(policy, root, flow_steps=8, sample_average=1, seed=42)


@pytest.mark.parametrize("where", ["loaded", "checkpoint"])
def test_rejects_n16_configuration(tmp_path: Path, where: str) -> None:
    root = make_checkpoint(tmp_path / "checkpoint")
    policy = make_policy()
    if where == "loaded":
        policy.model.config.model_type = "Gr00tN1d6"
    else:
        config = json.loads((root / "config.json").read_text())
        config["model_type"] = "Gr00tN1d6"
        (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not the audited"):
        build_served_identity(policy, root, flow_steps=8, sample_average=1, seed=42)


def test_checks_actual_flow_steps_and_sampling_expectations(tmp_path: Path) -> None:
    root = make_checkpoint(tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="actual loaded action head"):
        build_served_identity(make_policy(flow_steps=4), root, flow_steps=8, sample_average=1, seed=42)
    identity = make_identity(root)
    for key, wrong in (("expected_flow_steps", 4), ("expected_sample_average", 2), ("expected_seed", 43)):
        with pytest.raises(ValueError, match="requested value"):
            validate_model_identity(identity, **{key: wrong})
    for bad_steps in (True, 0, 4.5):
        with pytest.raises(ValueError, match="positive integers"):
            build_served_identity(make_policy(), root, flow_steps=bad_steps, sample_average=1, seed=42)


def test_tampered_identity_and_nested_checkpoint_are_rejected(tmp_path: Path) -> None:
    root = make_checkpoint(tmp_path / "checkpoint")
    identity = make_identity(root)
    modified = deepcopy(identity)
    modified["sampling"]["seed"] += 1
    with pytest.raises(ValueError, match="identity digest"):
        validate_model_identity(modified)
    modified = deepcopy(identity["checkpoint"])
    modified["files"]["statistics.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checkpoint identity digest"):
        validate_model_identity(identity, expected_checkpoint_identity=modified)


def test_action_binding_preserves_action_info_and_call_count() -> None:
    action = object()
    info = {"timing": 0.01}
    calls = []

    def get_action(observation: object) -> tuple[object, dict]:
        calls.append(observation)
        return action, info

    wrapped = bind_action_identity(get_action, "a" * 64)
    observation = object()
    output, bound_info = wrapped(observation)
    assert output is action
    assert calls == [observation]
    assert info == {"timing": 0.01}
    assert bound_info == {"timing": 0.01, "model_identity_id": "a" * 64}
    assert bound_info is not info


def test_action_binding_handles_none_and_rejects_conflicting_identity() -> None:
    action = object()
    output, info = bind_action_identity(lambda: (action, None), "a" * 64)()
    assert output is action
    assert info == {"model_identity_id": "a" * 64}
    with pytest.raises(ValueError, match="different model identity"):
        bind_action_identity(lambda: (action, {"model_identity_id": "b" * 64}), "a" * 64)()
    with pytest.raises(ValueError, match="mapping or None"):
        bind_action_identity(lambda: (action, [1]), "a" * 64)()
