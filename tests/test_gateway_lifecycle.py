from __future__ import annotations

import signal

import emissary_router.cli as cli
import emissary_router.launch as launch


def _boom(_args):
    raise RuntimeError("gateway exploded")


def test_main_turns_runtime_error_into_clean_exit(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_cmd_status", _boom)
    rc = cli.main(["status"])
    assert rc == 1
    assert "gateway exploded" in capsys.readouterr().err  # no raw traceback


def test_stop_gateway_escalates_to_sigkill(monkeypatch):
    sent: list[int] = []
    state = {"alive": True}
    monkeypatch.setattr(launch, "_read_pid", lambda: 4242)
    monkeypatch.setattr(launch, "_process_running", lambda pid: state["alive"])
    monkeypatch.setattr(launch, "_remove_pid_file", lambda: None)
    monkeypatch.setattr(launch, "_wait_for_exit", lambda pid, timeout: not state["alive"])

    def fake_kill(pid, sig):
        sent.append(sig)
        if sig == signal.SIGKILL:
            state["alive"] = False

    monkeypatch.setattr(launch.os, "kill", fake_kill)

    status = launch.stop_gateway()
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert status.message == "killed"
