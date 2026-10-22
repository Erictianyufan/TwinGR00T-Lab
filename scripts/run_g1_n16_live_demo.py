"""Run a real GR00T N1.6 G1 policy in Isaac Lab-Arena on one GPU.

The server is always started with ``--model-path``; the replay-policy
``--dataset-path`` route is deliberately unavailable in this launcher.
No scripted grasp, object teleport, or demo-assist option is exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_CLASS = (
    "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy."
    "Gr00tRemoteClosedloopPolicy"
)


def checkpoint_manifest(checkpoint: Path) -> dict[str, object]:
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "Gr00tN1d6":
        raise ValueError(
            f"Expected the official gn1_6 checkpoint, got model_type="
            f"{config.get('model_type')!r} in {config_path}"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (checkpoint / shard).is_file() for shard in shards):
        raise FileNotFoundError(f"Checkpoint shards are incomplete in {checkpoint}: {shards}")
    for name in ("processor_config.json", "embodiment_id.json", "statistics.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    return {
        "path": str(checkpoint),
        "revision": "gn1_6",
        "model_type": config["model_type"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "shards": {name: (checkpoint / name).stat().st_size for name in shards},
    }


def free_vram_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.policy_python),
        "-m", "gr00t.eval.run_gr00t_server",
        "--model-path", str(args.checkpoint),
        "--embodiment-tag", "NEW_EMBODIMENT",
        "--device", "cuda:0",
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]


def runner_command(args: argparse.Namespace) -> list[str]:
    config = args.arena_root / "isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml"
    command = [
        str(args.sim_python),
        "-m", "isaaclab_arena.evaluation.policy_runner",
        "--headless",
        "--device", "cuda:0",
        "--policy_type", POLICY_CLASS,
        "--policy_config_yaml_path", str(config),
        "--remote_host", "127.0.0.1",
        "--remote_port", str(args.port),
        "--num_steps", str(args.steps),
        "--seed", str(args.seed),
        "--enable_cameras",
        "--record_camera_video",
        "--output_base_dir", str(args.output_dir),
        "galileo_g1_locomanip_pick_and_place",
        "--object", "brown_box",
        "--embodiment", "g1_wbc_joint",
    ]
    if args.viewport_video:
        command.insert(command.index("--output_base_dir"), "--record_viewport_video")
    return command


def wait_for_server(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"GR00T server exited during model load (code {process.returncode})")
        if port_in_use(port):
            return
        time.sleep(2)
    raise TimeoutError(f"GR00T server did not listen on 127.0.0.1:{port} within {timeout}s")


def run(args: argparse.Namespace) -> int:
    args.arena_root = args.arena_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    for executable in (args.sim_python, args.policy_python):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    if not args.arena_root.is_dir():
        raise FileNotFoundError(args.arena_root)
    model = checkpoint_manifest(args.checkpoint)
    available = free_vram_mib()
    if available < args.min_free_vram_mib:
        raise RuntimeError(
            f"Only {available} MiB free on GPU 0; require at least "
            f"{args.min_free_vram_mib} MiB. Leave the other project running "
            "and schedule this demo for a free-GPU window; no process was stopped."
        )
    if port_in_use(args.port):
        raise RuntimeError(f"Port {args.port} is already in use; refusing to take over another service")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        "status": "starting",
        "controller_source": "GR00T_N1.6_model_inference",
        "replay_used": False,
        "scripted_assistance_used": False,
        "task": "galileo_g1_locomanip_pick_and_place",
        "checkpoint": model,
        "arena_root": str(args.arena_root),
        "initial_free_vram_mib": available,
        "server_command": server_command(args),
        "runner_command": runner_command(args),
        "started_at": datetime.now().isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    server = None
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with (args.output_dir / "server.log").open("wb") as server_log:
            server = subprocess.Popen(
                server_command(args), cwd=args.arena_root / "submodules/Isaac-GR00T",
                stdout=server_log, stderr=subprocess.STDOUT, env=env,
            )
            wait_for_server(server, args.port, args.server_timeout)
            manifest["status"] = "server_ready"
            manifest["free_vram_after_model_mib"] = free_vram_mib()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with (args.output_dir / "runner.log").open("wb") as runner_log:
                result = subprocess.run(
                    runner_command(args), cwd=args.arena_root,
                    stdout=runner_log, stderr=subprocess.STDOUT, env=env,
                    check=False,
                )
            manifest["runner_exit_code"] = result.returncode
            manifest["status"] = "rollout_finished" if result.returncode == 0 else "rollout_failed"
            manifest["video_files"] = [
                str(path) for path in sorted(args.output_dir.rglob("*.mp4"))
            ]
            manifest["episode_result_files"] = [
                str(path) for path in sorted(args.output_dir.rglob("episode_results*.jsonl"))
            ]
            return result.returncode
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        manifest["server_exit_code"] = server.returncode if server is not None else None
        manifest["finished_at"] = datetime.now().isoformat()
        manifest["final_free_vram_mib"] = free_vram_mib()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-root", type=Path, default=PROJECT_ROOT / "vendor/IsaacLab-Arena-0.3.0")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation-gn1_6",
    )
    parser.add_argument("--sim-python", type=Path, default=Path("/home/ubuntu/env_isaaclab/bin/python"))
    parser.add_argument("--policy-python", type=Path, default=Path("/home/ubuntu/env_twingroot_groot/bin/python"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results/g1_live" / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--min-free-vram-mib", type=int, default=37000)
    parser.add_argument("--server-timeout", type=float, default=600)
    parser.add_argument("--viewport-video", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
