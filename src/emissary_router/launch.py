from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from emissary_router.config import AppConfig, emissary_router_home


def state_dir() -> Path:
    return emissary_router_home()


def pid_path() -> Path:
    return state_dir() / "server.pid"


def server_log_path() -> Path:
    return state_dir() / "server.log"


def gateway_url(config: AppConfig) -> str:
    host = "127.0.0.1" if config.server.host in {"0.0.0.0", "::"} else config.server.host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{config.server.port}"


@dataclass(frozen=True)
class GatewayStatus:
    healthy: bool
    url: str
    pid: int | None
    message: str


def gateway_status(config: AppConfig) -> GatewayStatus:
    url = gateway_url(config)
    if _health_ok(url):
        return GatewayStatus(True, url, _read_pid(), "healthy")
    pid = _read_pid()
    if pid and _process_running(pid):
        return GatewayStatus(False, url, pid, "process running but health check failed")
    return GatewayStatus(False, url, pid, "stopped")


def ensure_gateway(config: AppConfig, config_path: Path, pricing_path: Path) -> GatewayStatus:
    status = gateway_status(config)
    if status.healthy:
        return status
    if status.pid and _process_running(status.pid):
        raise RuntimeError(
            f"emissary-router process {status.pid} is running but {status.url} is not healthy; "
            f"see {server_log_path()}"
        )

    state_dir().mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["EMISSARY_ROUTER_CONFIG"] = str(config_path)
    env["EMISSARY_ROUTER_PRICING"] = str(pricing_path)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "emissary_router.server:create_app",
        "--factory",
        "--host",
        config.server.host,
        "--port",
        str(config.server.port),
    ]
    log_handle = server_log_path().open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_handle.close()

    pid_path().write_text(str(proc.pid))
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"emissary-router exited early with code {proc.returncode}; see {server_log_path()}"
            )
        if _health_ok(status.url):
            return GatewayStatus(True, status.url, proc.pid, "started")
        time.sleep(0.2)
    raise RuntimeError(
        f"emissary-router did not become healthy at {status.url}; see {server_log_path()}"
    )


def stop_gateway() -> GatewayStatus:
    pid = _read_pid()
    if not pid:
        return GatewayStatus(False, "", None, "no pid file")
    if not _process_running(pid):
        _remove_pid_file()
        return GatewayStatus(False, "", pid, "stale pid file removed")

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 10
    while time.time() < deadline:
        if not _process_running(pid):
            _remove_pid_file()
            return GatewayStatus(False, "", pid, "stopped")
        time.sleep(0.2)
    return GatewayStatus(False, "", pid, "still running after SIGTERM")


def exec_claude(
    config: AppConfig,
    config_path: Path,
    pricing_path: Path,
    claude_command: str,
    claude_args: list[str],
    dry_run: bool = False,
) -> int:
    status = ensure_gateway(config, config_path, pricing_path)
    env = os.environ.copy()
    env["EMISSARY_ROUTER_CONFIG"] = str(config_path)
    env["EMISSARY_ROUTER_PRICING"] = str(pricing_path)
    env["ANTHROPIC_BASE_URL"] = status.url
    if config.server.auth_key:
        env["ANTHROPIC_API_KEY"] = config.server.auth_key
    env.setdefault("ENABLE_TOOL_SEARCH", "true")
    env.setdefault("CLAUDE_CODE_ATTRIBUTION_HEADER", "0")

    argv = [claude_command, *claude_args]
    if dry_run:
        print(f"emissary-router gateway: {status.url} ({status.message})")
        print("exec:", " ".join(argv))
        print("env ANTHROPIC_BASE_URL=" + env["ANTHROPIC_BASE_URL"])
        if config.server.auth_key:
            print("env ANTHROPIC_API_KEY=<Emissary Router auth key>")
        print("env ENABLE_TOOL_SEARCH=" + env["ENABLE_TOOL_SEARCH"])
        print("env CLAUDE_CODE_ATTRIBUTION_HEADER=" + env["CLAUDE_CODE_ATTRIBUTION_HEADER"])
        return 0

    try:
        os.execvpe(claude_command, argv, env)
    except FileNotFoundError:
        print(
            f"emissary-router: could not find Claude Code command '{claude_command}'",
            file=sys.stderr,
        )
        return 127
    return 0


def _health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/", timeout=0.75) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def _read_pid() -> int | None:
    try:
        return int(pid_path().read_text().strip())
    except (OSError, ValueError):
        return None


def _remove_pid_file() -> None:
    try:
        pid_path().unlink()
    except FileNotFoundError:
        pass


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
