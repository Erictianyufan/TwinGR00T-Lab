"""Guard against inferring action semantics from stale model config metadata."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "launch_aloha_n17_lowmem.py"
_SPEC = importlib.util.spec_from_file_location("aloha_lowmem_contract", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_processor_flag_wins_over_stale_model_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"use_relative_action": False}))
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"use_relative_action": True, "max_action_horizon": 40}})
    )
    contract = _MODULE.inspect_checkpoint_processor(str(tmp_path))
    assert contract["use_relative_action"] is True
    assert contract["source"] == str((processor / "processor_config.json").resolve())


def test_root_processor_has_same_precedence_as_policy_loader(tmp_path: Path) -> None:
    (tmp_path / "processor").mkdir()
    for relative, directory in ((True, tmp_path), (False, tmp_path / "processor")):
        (directory / "processor_config.json").write_text(
            json.dumps({"processor_kwargs": {"use_relative_action": relative}})
        )
    assert _MODULE.inspect_checkpoint_processor(str(tmp_path))["use_relative_action"] is True


def test_string_false_cannot_silently_become_truthy(tmp_path: Path) -> None:
    (tmp_path / "processor_config.json").write_text(json.dumps({"processor_kwargs": {"use_relative_action": "false"}}))
    with pytest.raises(ValueError, match="JSON boolean"):
        _MODULE.inspect_checkpoint_processor(str(tmp_path))


def test_uninspected_hub_checkpoint_does_not_claim_a_verified_contract() -> None:
    result = _MODULE.inspect_checkpoint_processor("nvidia/GR00T-N1.7-3B")
    assert result["status"] == "remote_not_inspected"
    assert result["use_relative_action"] is None


def test_default_visual_mode_leaves_existing_configuration_unchanged() -> None:
    inherited = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
    model = SimpleNamespace(color_jitter_params=inherited, crop_fraction=0.95, random_rotation_angle=5)
    before = vars(model).copy()
    summary = _MODULE.configure_visual_augmentation(model, "inherit", {})
    assert vars(model) == before
    assert model.color_jitter_params is inherited
    assert summary["color_jitter_override"] is None
    assert "not verified by dry-run" in summary["verification"]


def test_color_off_uses_explicit_zero_override_without_claiming_crop_disabled() -> None:
    model = SimpleNamespace(color_jitter_params=None, crop_fraction=0.95, random_rotation_angle=5)
    source = {"visual_augmentation": {"crop_fraction": 0.95, "random_rotation_angle": 5}}
    summary = _MODULE.configure_visual_augmentation(model, "color-off", source)
    assert model.color_jitter_params == {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0}
    assert (model.crop_fraction, model.random_rotation_angle) == (0.95, 5)
    assert summary["crop_and_rotation"] == "inherited_not_overridden"
    assert summary["starting_checkpoint_visual_config"] == source["visual_augmentation"]


def test_visual_audit_uses_effective_processor_not_model_config(tmp_path: Path) -> None:
    zero_jitter = dict.fromkeys(("brightness", "contrast", "saturation", "hue"), 0.0)
    (tmp_path / "config.json").write_text(json.dumps({"color_jitter_params": zero_jitter}))
    processor = tmp_path / "processor"
    processor.mkdir()
    config_path = processor / "processor_config.json"
    config_path.write_text(json.dumps({"processor_kwargs": {"color_jitter_params": {**zero_jitter, "hue": 0.08}}}))
    with pytest.raises(RuntimeError, match="did not reach the saved effective processor"):
        _MODULE.verify_saved_visual_augmentation(tmp_path, "color-off")

    config_path.write_text(
        json.dumps({"processor_kwargs": {"color_jitter_params": zero_jitter, "crop_fraction": 0.95}})
    )
    audit = _MODULE.verify_saved_visual_augmentation(tmp_path, "color-off")
    assert audit["status"] == "verified_color_off"
    assert audit["effective_visual_config"]["crop_fraction"] == 0.95
    assert audit["effective_processor_source"] == str(config_path.resolve())


@pytest.mark.parametrize("jitter", [None, {}, {"brightness": 0.0}, {"brightness": float("nan")}])
def test_visual_audit_rejects_missing_or_partial_zero_overrides(tmp_path: Path, jitter: object) -> None:
    (tmp_path / "processor_config.json").write_text(json.dumps({"processor_kwargs": {"color_jitter_params": jitter}}))
    with pytest.raises(RuntimeError, match="color-off did not reach"):
        _MODULE.verify_saved_visual_augmentation(tmp_path, "color-off")
    assert _MODULE.verify_saved_visual_augmentation(tmp_path, "inherit")["status"] == "inherit_recorded"
