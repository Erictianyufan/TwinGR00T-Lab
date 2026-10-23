"""Guard against inferring action semantics from stale model config metadata."""

import importlib.util
import json
from pathlib import Path

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
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"use_relative_action": "false"}})
    )
    with pytest.raises(ValueError, match="JSON boolean"):
        _MODULE.inspect_checkpoint_processor(str(tmp_path))


def test_uninspected_hub_checkpoint_does_not_claim_a_verified_contract() -> None:
    result = _MODULE.inspect_checkpoint_processor("nvidia/GR00T-N1.7-3B")
    assert result["status"] == "remote_not_inspected"
    assert result["use_relative_action"] is None
