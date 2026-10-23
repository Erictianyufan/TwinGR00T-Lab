"""CPU-only, content-addressed identity for the N1.7 policy service.

This binds a service's declared loaded model and sampling settings to checkpoint
file contents. It is reproducibility evidence, not a signature, remote hardware
attestation, proof of individual tensor contents in GPU memory, or task success.
Only allowlisted public configuration fields are exposed; never dump configs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

_MODEL_TYPE = "Gr00tN1d7"
_BACKBONE = "nvidia/Cosmos-Reason2-2B"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: Mapping[str, Any], id_key: str) -> str:
    payload = {key: item for key, item in value.items() if key != id_key}
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Model identity must contain finite JSON-compatible values") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _relative_path(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Checkpoint filenames must be safe relative POSIX paths")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
        raise ValueError("Checkpoint filenames must be safe relative POSIX paths")
    return name


def _file(root: Path, name: str) -> Path:
    path = root / _relative_path(name)
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Checkpoint file missing or outside checkpoint directory: {name}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint JSON must be an object: {path.name}")
    return value


def _model_metadata(config: Any) -> dict[str, Any]:
    def get(key: str) -> Any:
        return config.get(key) if isinstance(config, Mapping) else getattr(config, key, None)

    metadata = {
        "model_type": get("model_type"),
        "architectures": get("architectures"),
        "backbone_model": get("model_name"),
    }
    _validate_model_metadata(metadata)
    return metadata


def _validate_model_metadata(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Missing N1.7 model metadata")
    # This service intentionally supports the audited N1.7/Cosmos-Reason2 setup.
    if value.get("model_type") != _MODEL_TYPE or value.get("architectures") != [_MODEL_TYPE]:
        raise ValueError("Loaded/checkpoint architecture is not the audited GR00T N1.7 model")
    if value.get("backbone_model") != _BACKBONE:
        raise ValueError("Loaded/checkpoint backbone is not the audited Cosmos-Reason2-2B model")


def _fingerprint(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """Hash complete model shards and the effective processor/config files.

    Root processor files take precedence, matching Gr00tPolicy. Files must be
    completed before calling; concurrent changes are rejected. No GPU imports.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError("Checkpoint directory does not exist")
    config_path = _file(root, "config.json")
    config_snapshot = _fingerprint(config_path)
    metadata = _model_metadata(_read_json(config_path))
    processor_name = "processor_config.json"
    if not (root / processor_name).is_file():
        processor_name = "processor/processor_config.json"
    processor_dir = PurePosixPath(processor_name).parent
    names = {"config.json", processor_name}
    for name in ("statistics.json", "embodiment_id.json"):
        names.add((processor_dir / name).as_posix())

    index_name = "model.safetensors.index.json"
    index_snapshot = None
    if (root / index_name).is_file():
        index_path = _file(root, index_name)
        index_snapshot = _fingerprint(index_path)
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Checkpoint shard index has no weight_map")
        names.add(index_name)
        for name in weight_map.values():
            name = _relative_path(name)
            if not name.endswith(".safetensors"):
                raise ValueError("Checkpoint index references a non-safetensors shard")
            names.add(name)
    else:
        names.add("model.safetensors")

    paths = {name: _file(root, name) for name in sorted(names)}
    before = {name: _fingerprint(file) for name, file in paths.items()}
    if before["config.json"] != config_snapshot:
        raise ValueError("Checkpoint config changed while reading identity")
    if index_snapshot is not None and before[index_name] != index_snapshot:
        raise ValueError("Checkpoint index changed while reading identity")
    files = {}
    for name, file in paths.items():
        if before[name][0] <= 0:
            raise ValueError(f"Checkpoint file is empty: {name}")
        if name.endswith(".json"):
            _read_json(file)
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[name] = {"sha256": digest, "size_bytes": before[name][0]}
    if any(_fingerprint(file) != before[name] for name, file in paths.items()):
        raise ValueError("Checkpoint changed while hashing; retry only after saving completes")
    result = {
        "schema_version": 1,
        "model_config": metadata,
        "processor_config_path": processor_name,
        "files": files,
    }
    result["checkpoint_id"] = _digest(result, "checkpoint_id")
    _validate_checkpoint_identity(result)
    return result


def _validate_checkpoint_identity(value: Any) -> None:
    if not isinstance(value, Mapping) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Invalid checkpoint identity schema")
    if not _is_digest(value.get("checkpoint_id")) or value["checkpoint_id"] != _digest(value, "checkpoint_id"):
        raise ValueError("Checkpoint identity digest mismatch")
    _validate_model_metadata(value.get("model_config"))
    processor_name = value.get("processor_config_path")
    if processor_name not in ("processor_config.json", "processor/processor_config.json"):
        raise ValueError("Checkpoint identity has invalid effective processor path")
    files = value.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Checkpoint identity has no file hashes")
    for name, item in files.items():
        _relative_path(name)
        if not isinstance(item, Mapping) or not _is_digest(item.get("sha256")):
            raise ValueError("Invalid checkpoint file SHA256")
        if type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0:
            raise ValueError("Invalid checkpoint file size")
    required = {"config.json", processor_name}
    required.update((PurePosixPath(processor_name).parent / name).as_posix()
                    for name in ("statistics.json", "embodiment_id.json"))
    if not required.issubset(files):
        raise ValueError("Checkpoint identity is missing configuration/statistics file hashes")
    shards = [name for name in files if name.endswith(".safetensors")]
    if not shards or ("model.safetensors" not in files and "model.safetensors.index.json" not in files):
        raise ValueError("Checkpoint identity is missing model weight file hashes")


def _sampling(flow_steps: Any, sample_average: Any, seed: Any) -> dict[str, int]:
    if type(flow_steps) is not int or flow_steps <= 0 or type(sample_average) is not int or sample_average <= 0:
        raise ValueError("flow_steps and sample_average must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return {"flow_steps": flow_steps, "sample_average": sample_average, "seed": seed}


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def build_served_identity(
    policy: Any, path: str | Path, *, flow_steps: int, sample_average: int, seed: int,
) -> dict[str, Any]:
    """Inspect an already-loaded policy and bind it to checkpoint contents."""
    sampling = _sampling(flow_steps, sample_average, seed)
    model = getattr(policy, "model", None)
    processor = getattr(policy, "processor", None)
    if model is None or processor is None:
        raise ValueError("Policy must have an actual loaded model and processor")
    if type(model).__name__ != _MODEL_TYPE:
        raise ValueError("Actual loaded model class is not Gr00tN1d7")
    metadata = _model_metadata(getattr(model, "config", None))
    if getattr(getattr(model, "action_head", None), "num_inference_timesteps", None) != flow_steps:
        raise ValueError("Requested flow_steps differ from the actual loaded action head")
    embodiment = getattr(policy, "embodiment_tag", None)
    embodiment = getattr(embodiment, "value", embodiment)
    if embodiment != "new_embodiment":
        raise ValueError("ALOHA policy identity requires new_embodiment")
    checkpoint = checkpoint_identity(path)
    if metadata != checkpoint["model_config"]:
        raise ValueError("Loaded model configuration differs from checkpoint")
    result = {
        "schema_version": 1,
        "model_family": "GR00T_N1.7",
        "loaded_model": {
            "policy_class": _class_name(policy),
            "model_class": _class_name(model),
            "processor_class": _class_name(processor),
            **metadata,
            "embodiment_tag": embodiment,
        },
        "checkpoint": checkpoint,
        "sampling": sampling,
    }
    result["identity_id"] = _digest(result, "identity_id")
    validate_model_identity(result)
    return result


def validate_model_identity(
    identity: Any,
    *,
    expected_checkpoint_identity: Mapping[str, Any] | None = None,
    expected_flow_steps: int | None = None,
    expected_sample_average: int | None = None,
    expected_seed: int | None = None,
) -> None:
    """Reject internally inconsistent identities and mismatched local expectations.

    Supplying a freshly computed expected_checkpoint_identity verifies contents
    against the caller's checkpoint, not merely matching a service's own label.
    """
    if not isinstance(identity, Mapping):
        raise ValueError("Model identity must be an object")
    if type(identity.get("schema_version")) is not int or identity["schema_version"] != 1:
        raise ValueError("Unsupported model identity schema")
    if not _is_digest(identity.get("identity_id")) or identity["identity_id"] != _digest(identity, "identity_id"):
        raise ValueError("Model identity digest mismatch")
    if identity.get("model_family") != "GR00T_N1.7":
        raise ValueError("Service model family is not GR00T N1.7")
    loaded = identity.get("loaded_model")
    _validate_model_metadata(loaded)
    for key in ("policy_class", "model_class", "processor_class"):
        if not isinstance(loaded.get(key), str) or not loaded[key]:
            raise ValueError("Missing actual loaded class identity")
    if loaded["model_class"].rsplit(".", 1)[-1] != _MODEL_TYPE:
        raise ValueError("Loaded model class is not Gr00tN1d7")
    if loaded.get("embodiment_tag") != "new_embodiment":
        raise ValueError("Service embodiment is not ALOHA new_embodiment")
    checkpoint = identity.get("checkpoint")
    _validate_checkpoint_identity(checkpoint)
    if any(loaded[key] != value for key, value in checkpoint["model_config"].items()):
        raise ValueError("Loaded/checkpoint model metadata mismatch")
    if expected_checkpoint_identity is not None:
        _validate_checkpoint_identity(expected_checkpoint_identity)
        if checkpoint != expected_checkpoint_identity:
            raise ValueError("Service checkpoint content identity differs from local expected checkpoint")
    sample = identity.get("sampling")
    if not isinstance(sample, Mapping):
        raise ValueError("Missing sampling identity")
    _sampling(sample.get("flow_steps"), sample.get("sample_average"), sample.get("seed"))
    for key, expected in (
        ("flow_steps", expected_flow_steps), ("sample_average", expected_sample_average), ("seed", expected_seed),
    ):
        if expected is not None and (type(expected) is not int or sample[key] != expected):
            raise ValueError(f"Service {key} differs from the requested value")


def bind_action_identity(get_action: Callable[..., Any], identity_id: str) -> Callable[..., Any]:
    """Add response identity without copying, transforming, or replacing action."""
    if not _is_digest(identity_id):
        raise ValueError("Invalid model identity ID")

    def identified_get_action(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        action, info = get_action(*args, **kwargs)
        if info is not None and not isinstance(info, Mapping):
            raise ValueError("Policy get_action info must be a mapping or None")
        bound_info = dict(info or {})
        previous = bound_info.get("model_identity_id")
        if previous is not None and previous != identity_id:
            raise ValueError("Policy response already has a different model identity")
        bound_info["model_identity_id"] = identity_id
        return action, bound_info

    return identified_get_action
