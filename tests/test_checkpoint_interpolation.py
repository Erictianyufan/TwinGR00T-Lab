"""Small checkpoint tests for the experimental N1.7 parameter interpolation."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
interpolate = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/interpolate_n17_checkpoints.py"))[
    "interpolate"
]


def checkpoint(path: Path, scalar: float) -> None:
    path.mkdir()
    filename = "model-00001-of-00001.safetensors"
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 16}, "weight_map": {"float": filename, "integer": filename}}),
        encoding="utf-8",
    )
    (path / "config.json").write_text("{}", encoding="utf-8")
    safetensors_torch.save_file(
        {"float": torch.tensor([scalar, scalar], dtype=torch.float32), "integer": torch.tensor([1])},
        path / filename,
    )


def test_interpolation_is_exact_and_does_not_modify_sources(tmp_path: Path) -> None:
    first, second, output = (tmp_path / name for name in ("first", "second", "output"))
    checkpoint(first, 1.0)
    checkpoint(second, 3.0)
    result = interpolate(first, second, output, 0.75)
    mixed = safetensors_torch.load_file(output / "model-00001-of-00001.safetensors")
    original = safetensors_torch.load_file(first / "model-00001-of-00001.safetensors")
    assert torch.equal(mixed["float"], torch.tensor([2.5, 2.5]))
    assert torch.equal(mixed["integer"], torch.tensor([1]))
    assert torch.equal(original["float"], torch.tensor([1.0, 1.0]))
    assert result["status"] == "experimental_parameter_interpolation_not_task_validation"
    assert json.loads((output / "interpolation.json").read_text())["alpha_second"] == 0.75
    with pytest.raises(FileExistsError, match="overwrite"):
        interpolate(first, second, output, 0.5)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_invalid_alpha_rejected(tmp_path: Path, alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        interpolate(tmp_path / "first", tmp_path / "second", tmp_path / "out", alpha)
