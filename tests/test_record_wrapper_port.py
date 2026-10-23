"""Exercise the real shell wrapper with CPU-only, isolated command shims."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def bash_executable():
    candidates = [os.environ.get("TWINGROOT_TEST_BASH")]
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidates.append(str(Path(git).resolve().parents[1] / "bin/bash.exe"))
    else:
        candidates.append(shutil.which("bash"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("Bash is required for the wrapper integration test")


@pytest.fixture
def wrapper_environment(tmp_path, bash_executable):
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()

    def shim(name, contents):
        path = shim_dir / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + contents, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        return path.as_posix()

    shim(
        "ss",
        """printf '%s\n' "$*" >> "$SS_LOG"
count=0
[[ ! -f "$SS_COUNT" ]] || read -r count < "$SS_COUNT"
count=$((count+1))
printf '%s\n' "$count" > "$SS_COUNT"
if [[ "$SS_MODE" == query-error ]]; then echo 'permission failure' >&2; exit 2; fi
if [[ "$SS_MODE" == occupied ]]; then
  echo 'LISTEN 0 128 127.0.0.1:5555 0.0.0.0:* users:(("existing",pid=999999,fd=6))'
  exit 0
fi
if [[ "$count" == 1 || ! -f "$SERVER_PID_FILE" ]]; then exit 0; fi
read -r owner < "$SERVER_PID_FILE"
case "$SS_MODE" in
  own) ;;
  foreign) owner=999999 ;;
  pid-prefix) owner="${owner}0" ;;
  hidden) echo 'LISTEN 0 128 127.0.0.1:5555 0.0.0.0:*'; exit 0 ;;
  later-query-error) echo 'query failed after start' >&2; exit 2 ;;
esac
printf 'LISTEN 0 128 127.0.0.1:5555 0.0.0.0:* users:(("python",pid=%s,fd=6))\n' "$owner"
if [[ "$SS_MODE" == mixed-owners ]]; then
  echo 'LISTEN 0 128 [::]:5555 [::]:* users:(("other",pid=999999,fd=7))'
fi
""",
    )
    shim("sleep", "exec /usr/bin/sleep 0.01\n")
    server = shim(
        "server",
        """printf '%s\n' "$$" > "$SERVER_PID_FILE"
trap 'printf terminated > "$SERVER_STOPPED"; exit 0' TERM
while :; do /usr/bin/sleep 0.05; done
""",
    )
    simulator = shim("simulator", 'printf invoked > "$SIM_MARKER"\n')
    model = tmp_path / "checkpoint"
    model.mkdir()
    demo = tmp_path / "demo.hdf5"
    demo.touch()
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(shim_dir) + os.pathsep + env["PATH"],
            "MODEL_PATH": model.as_posix(),
            "GROOT_ROOT": tmp_path.as_posix(),
            "GROOT_PYTHON": server,
            "SIM_PYTHON": simulator,
            "HELDOUT_DEMO": demo.as_posix(),
            "OUTPUT_VIDEO": (tmp_path / "result.mp4").as_posix(),
            "SS_LOG": (tmp_path / "ss.log").as_posix(),
            "SS_COUNT": (tmp_path / "ss.count").as_posix(),
            "SERVER_PID_FILE": (tmp_path / "server.pid").as_posix(),
            "SERVER_STOPPED": (tmp_path / "server.stopped").as_posix(),
            "SIM_MARKER": (tmp_path / "sim.invoked").as_posix(),
            "POLICY_PORT": "5555",
            "SCENE_STYLE": "factory",
        }
    )
    wrapper = Path(__file__).resolve().parents[1] / "scripts/record_aloha_n17_factory.sh"

    def run(mode, **overrides):
        return subprocess.run(
            [bash_executable, wrapper.as_posix()],
            env={**env, "SS_MODE": mode, **overrides},
            capture_output=True,
            text=True,
            timeout=20,
        )

    return tmp_path, run


@pytest.mark.parametrize("mode,message", [("occupied", "already occupied"), ("query-error", "Cannot inspect")])
def test_existing_or_uninspectable_port_never_starts_a_server(wrapper_environment, mode, message):
    folder, run = wrapper_environment
    result = run(mode)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (folder / "server.pid").exists()
    assert not (folder / "server.stopped").exists()
    assert not (folder / "sim.invoked").exists()


@pytest.mark.parametrize("mode", ["foreign", "pid-prefix", "hidden", "mixed-owners", "later-query-error"])
def test_foreign_hidden_or_racing_listener_is_never_ready(wrapper_environment, mode):
    folder, run = wrapper_environment
    result = run(mode)
    assert result.returncode != 0
    assert "refusing to connect" in result.stderr or "Cannot inspect policy socket" in result.stderr
    assert (folder / "server.pid").exists()
    assert (folder / "server.stopped").read_text() == "terminated"
    assert not (folder / "sim.invoked").exists()


def test_only_this_exact_process_listener_allows_simulation(wrapper_environment):
    folder, run = wrapper_environment
    result = run("own")
    assert result.returncode == 0, result.stderr
    assert (folder / "sim.invoked").read_text() == "invoked"
    assert (folder / "server.stopped").read_text() == "terminated"
    assert all(line == "-H -ltnp ( sport = :5555 )" for line in (folder / "ss.log").read_text().splitlines())


@pytest.mark.parametrize("port", ["0", "65536", "oops", "5555 ) or sport = :22"])
def test_invalid_port_fails_before_socket_query(wrapper_environment, port):
    folder, run = wrapper_environment
    result = run("own", POLICY_PORT=port)
    assert result.returncode != 0
    assert "POLICY_PORT must be" in result.stderr
    assert not (folder / "ss.log").exists()
    assert not (folder / "server.pid").exists()
