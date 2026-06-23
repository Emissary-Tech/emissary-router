from __future__ import annotations

import argparse
import getpass
import sys

from emissary_router.cli import _cmd_init
from emissary_router.config import read_env_file, user_config_path, user_env_path


def _prep(tmp_path, monkeypatch):
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    monkeypatch.delenv("EMISSARY_ROUTER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # avoid picking up a repo-root .env
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)


def test_init_skips_env_keys_and_writes_prompted(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    # ANTHROPIC is exported in the shell -> should be detected, not prompted/persisted
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    monkeypatch.delenv("EMISSARY_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    answers = iter(["emkey", "orkey"])  # EMISSARY_ROUTER_API_KEY, OPENROUTER_API_KEY
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": next(answers))

    assert _cmd_init(argparse.Namespace(no_prompt=False)) == 0
    assert user_config_path().exists()
    env = read_env_file(user_env_path())
    assert env["EMISSARY_ROUTER_API_KEY"] == "emkey"
    assert env["OPENROUTER_API_KEY"] == "orkey"
    assert "ANTHROPIC_API_KEY" not in env  # came from the shell, not persisted


def test_init_keeps_existing_value_on_empty_input(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    for key in ["EMISSARY_ROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    env_path = user_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("EMISSARY_ROUTER_API_KEY=old-key\n")
    # empty for EMISSARY (keep old), new values for the rest
    answers = iter(["", "ant", "orr"])
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": next(answers))

    assert _cmd_init(argparse.Namespace(no_prompt=False)) == 0
    env = read_env_file(env_path)
    assert env["EMISSARY_ROUTER_API_KEY"] == "old-key"  # kept
    assert env["ANTHROPIC_API_KEY"] == "ant"
    assert env["OPENROUTER_API_KEY"] == "orr"


def test_init_non_interactive_writes_template(tmp_path, monkeypatch):
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    monkeypatch.delenv("EMISSARY_ROUTER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    assert _cmd_init(argparse.Namespace(no_prompt=False)) == 0
    assert user_config_path().exists()
    assert user_env_path().exists()


def test_init_no_prompt_adds_missing_required_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    monkeypatch.delenv("EMISSARY_ROUTER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    for key in ["EMISSARY_ROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    env_path = user_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("EMISSARY_ROUTER_API_KEY=mykey\n")

    assert _cmd_init(argparse.Namespace(no_prompt=True)) == 0
    env = read_env_file(env_path)
    assert env["EMISSARY_ROUTER_API_KEY"] == "mykey"  # preserved
    assert env["ANTHROPIC_API_KEY"] == ""  # placeholder added
    assert env["OPENROUTER_API_KEY"] == ""


def test_init_backs_up_legacy_config_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    monkeypatch.delenv("EMISSARY_ROUTER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("models: {}\n")  # old yaml

    assert _cmd_init(argparse.Namespace(no_prompt=True)) == 0
    assert not (tmp_path / "config.yaml").exists()
    assert (tmp_path / "config.yaml.bak").exists()
    assert (tmp_path / "config.json").exists()
    assert user_config_path() == tmp_path / "config.json"


def test_init_via_main_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    monkeypatch.delenv("EMISSARY_ROUTER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    from emissary_router.cli import main

    assert main(["init", "--no-prompt"]) == 0
    assert user_config_path().exists()
    assert user_env_path().exists()
