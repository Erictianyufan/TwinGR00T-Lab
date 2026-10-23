"""Offline errors must be attributed to the actual episode language condition."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def offline():
    spec = importlib.util.spec_from_file_location(
        "offline_prompt_test",
        Path(__file__).resolve().parents[1] / "scripts/diagnose_n17_offline.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def episodes(tmp_path):
    demos = {}
    for split, text in (("train", "Use left, right, then left arm."), ("heldout", "Use right, left, then left arm.")):
        root = tmp_path / split
        (root / "data").mkdir(parents=True)
        (root / "instructions").mkdir()
        (root / "instructions/demo_0.json").write_text(json.dumps({"seen": [text, "Not the selected prompt"]}))
        demos[split] = root / "data/demo_0.hdf5"
    return demos


def test_each_episode_uses_its_own_first_seen_instruction(offline, episodes):
    selected = {split: offline.resolve_episode_prompt(demo, split=split) for split, demo in episodes.items()}
    assert selected["train"]["text"] == "Use left, right, then left arm."
    assert selected["heldout"]["text"] == "Use right, left, then left arm."
    for split in selected:
        assert selected[split]["source"] == "episode_instructions:seen[0]"
        assert selected[split]["instruction_path"] == str(
            (episodes[split].parent.parent / "instructions/demo_0.json").resolve(),
        )


def test_explicit_shared_override_remains_supported_and_labelled(offline, episodes):
    for split, demo in episodes.items():
        result = offline.resolve_episode_prompt(demo, split=split, shared_prompt="Same condition")
        assert result == {
            "text": "Same condition",
            "source": "--prompt (explicit shared override)",
            "instruction_path": None,
        }


def test_specific_override_wins_and_does_not_require_source_file(offline, tmp_path):
    result = offline.resolve_episode_prompt(
        tmp_path / "missing.hdf5",
        split="heldout",
        specific_prompt="Right arm",
        shared_prompt="Shared",
    )
    assert result == {"text": "Right arm", "source": "--heldout-prompt", "instruction_path": None}


@pytest.mark.parametrize("payload", [{}, {"seen": []}, {"seen": [None]}, {"seen": ["  "]}, {"seen": "not a list"}])
def test_missing_or_invalid_seen_is_not_silently_replaced(offline, episodes, payload):
    path = episodes["heldout"].parent.parent / "instructions/demo_0.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        offline.resolve_episode_prompt(episodes["heldout"], split="heldout")


@pytest.mark.parametrize("kwargs", [{"specific_prompt": " "}, {"shared_prompt": ""}])
def test_explicit_empty_override_is_an_error(offline, episodes, kwargs):
    with pytest.raises(ValueError, match="nonempty"):
        offline.resolve_episode_prompt(episodes["train"], split="train", **kwargs)


def test_main_passes_distinct_language_conditions_and_reports_dense_indices(
    offline,
    episodes,
    monkeypatch,
    capsys,
):
    calls = []

    def evaluate(policy, demo, prompt, indices, **kwargs):
        calls.append((demo, prompt, indices))
        return []

    monkeypatch.setattr(offline, "evaluate_episode", evaluate)
    monkeypatch.setattr(offline, "configure_policy_seed", lambda seed: {"policy_seed": seed})
    monkeypatch.setattr(offline, "configure_n17_sampling", lambda policy, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "gr00t.data.embodiment_tags",
        SimpleNamespace(EmbodimentTag=SimpleNamespace(NEW_EMBODIMENT="new_embodiment")),
    )
    monkeypatch.setitem(
        sys.modules, "gr00t.policy.gr00t_policy", SimpleNamespace(Gr00tPolicy=lambda **kwargs: object())
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_n17_offline.py",
            "--checkpoint",
            "unused",
            "--train-demo",
            str(episodes["train"]),
            "--heldout-demo",
            str(episodes["heldout"]),
            "--indices",
            "40",
            "41",
            "42",
            "400",
        ],
    )
    offline.main()
    report = json.loads(capsys.readouterr().out)
    assert calls == [
        (episodes["train"], "Use left, right, then left arm.", [40, 41, 42, 400]),
        (episodes["heldout"], "Use right, left, then left arm.", [40, 41, 42, 400]),
    ]
    assert report["shared_prompt_override_requested"] is False
    assert report["prompts"]["train"]["source"] == "episode_instructions:seen[0]"
    assert report["prompts"]["heldout"]["source"] == "episode_instructions:seen[0]"


def test_posttrain_offline_invocation_does_not_reuse_online_prompt():
    source = (Path(__file__).resolve().parents[1] / "scripts/run_n17_posttrain_checks.sh").read_text(encoding="utf-8")
    invocation = source.split('"${groot_python}" scripts/diagnose_n17_offline.py', 1)[1].split("failed=0", 1)[0]
    assert "--prompt" not in invocation
    assert "--train-demo" in invocation and "--heldout-demo" in invocation
