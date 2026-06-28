from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

import uvicorn

from emissary_router.catalog import CATALOG, ROUTER_API_KEY_ENV, cost_score
from emissary_router.config import (
    emissary_router_home,
    ensure_user_config,
    load_config,
    missing_runtime_env,
    read_env_file,
    user_config_path,
    user_env_path,
    write_env_file,
)
from emissary_router.launch import (
    announce_dashboard,
    announce_demo,
    dashboard_url,
    demo_url,
    ensure_gateway,
    exec_claude,
    gateway_status,
    stop_gateway,
)


def _cmd_config_path(_: argparse.Namespace) -> int:
    print(user_config_path())
    return 0


def _mask(value: str) -> str:
    return "****" + value[-4:] if len(value) > 4 else "****"


def _cmd_init(args: argparse.Namespace) -> int:
    print(f"Emissary Router setup -> {emissary_router_home()}")
    legacy_yaml = emissary_router_home() / "config.yaml"
    if legacy_yaml.exists():
        backup = legacy_yaml.parent / "config.yaml.bak"
        legacy_yaml.rename(backup)
        print(f"  config.yaml is no longer used; backed up to {backup.name}, using config.json")
    created = ensure_user_config()
    config_path = user_config_path()
    if created:
        print(f"  created {config_path.name}")

    shell_env = set(os.environ)  # snapshot real exports before .env is merged
    config = load_config(config_path)
    required = [ROUTER_API_KEY_ENV, *config.required_provider_env().values()]
    env_path = user_env_path()
    existing = read_env_file(env_path)

    interactive = sys.stdin.isatty() and not args.no_prompt
    if not interactive:
        values = dict(existing)
        added = False
        for key in required:
            if key not in values and key not in shell_env:
                values[key] = ""  # placeholder for the user to fill in
                added = True
        if added or not env_path.exists():
            write_env_file(env_path, values)
            print(f"  wrote {env_path}")
        missing = [k for k in required if not values.get(k) and k not in shell_env]
        if missing:
            print("Set these keys (edit the .env above or export them): " + ", ".join(missing))
        print("Then run: er code")
        return 0

    values = dict(existing)
    for key in required:
        if key in shell_env:
            print(f"  {key}: found in environment ✓")
            continue
        current = existing.get(key)
        if current:
            entry = getpass.getpass(f"  {key} [{_mask(current)}] (enter to keep): ")
            values[key] = entry or current
        else:
            entry = getpass.getpass(f"  {key}: ")
            if entry:
                values[key] = entry
    write_env_file(env_path, values)
    print(f"✓ wrote {env_path} (chmod 600)")
    print("Next: er code -- [claude args]")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None, strict_env=False)
    enabled = set(config.enabled_models())
    # Cheapest first, derived from pricing (the order routing scans).
    ordered = sorted(CATALOG.items(), key=lambda item: cost_score(item[1]))
    rows = []
    for name, spec in ordered:
        on = name in enabled
        resolved = config.resolve_model(name) if on else None
        rows.append(
            {
                "name": name,
                "enabled": on,
                "provider": resolved.provider if resolved else None,
                "model_id": resolved.model_id if resolved else None,
                "supported_providers": sorted(spec.providers),
                "default_provider": spec.default_provider,
                "cost_score": round(cost_score(spec), 4),
                "default": name == config.default,
            }
        )
    print(json.dumps({"models": rows}, indent=2))
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else user_config_path()
    config = load_config(config_path, strict_env=False)
    missing_env = missing_runtime_env(config)
    result = {
        "ok": not missing_env,
        "config": str(config_path.expanduser()),
        "default": config.default,
        "confidence": config.confidence,
        "enabled_models": config.enabled_models(),
        "required_provider_env": config.required_provider_env(),
        "missing_env": missing_env,
    }
    print(json.dumps(result, indent=2))
    return 0 if not missing_env else 2


def _load_config_or_hint(config_path: Path):
    try:
        return load_config(config_path)
    except FileNotFoundError:
        print(f"No config at {config_path}. Run `er init` to set up.", file=sys.stderr)
        return None


def _warn_missing_env(config) -> None:
    # Only the providers actually used by enabled models are required, so an
    # OpenRouter-only setup will not ask for an Anthropic key.
    missing = missing_runtime_env(config)
    if missing:
        print(
            "Warning: missing API keys for enabled models: " + ", ".join(missing)
            + " — requests will fail until these are set (.env or environment; "
            "run `er validate-config` to check).",
            file=sys.stderr,
        )


def _cmd_code(args: argparse.Namespace) -> int:
    _set_config_env(args.config)
    config_path = (Path(args.config) if args.config else user_config_path()).expanduser().resolve()
    config = _load_config_or_hint(config_path)
    if config is None:
        return 1
    _warn_missing_env(config)
    claude_args = list(args.claude_args)
    if claude_args and claude_args[0] == "--":
        claude_args = claude_args[1:]
    return exec_claude(
        config=config,
        config_path=config_path,
        claude_command=args.claude_command,
        claude_args=claude_args,
        dry_run=args.dry_run,
        open_dashboard=not args.no_open,
    )


def _cmd_debug(args: argparse.Namespace) -> int:
    _set_config_env(args.config)
    config = load_config(Path(args.config) if args.config else None)
    uvicorn.run(
        "emissary_router.server:create_app",
        host=config.server.host,
        port=config.server.port,
        factory=True,
    )
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    _set_config_env(args.config)
    config_path = (Path(args.config) if args.config else user_config_path()).expanduser().resolve()
    config = _load_config_or_hint(config_path)
    if config is None:
        return 1
    _warn_missing_env(config)
    status = ensure_gateway(config, config_path)
    print(json.dumps(status.__dict__, indent=2))
    announce_dashboard(config, status, open_browser=not getattr(args, "no_open", False))
    return 0 if status.healthy else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    _set_config_env(args.config)
    config_path = (Path(args.config) if args.config else user_config_path()).expanduser().resolve()
    config = _load_config_or_hint(config_path)
    if config is None:
        return 1
    if not config.demo.enabled:
        print('demo is disabled; set "demo": {"enabled": true} in your config', file=sys.stderr)
        return 1
    _warn_missing_env(config)
    status = ensure_gateway(config, config_path)
    print(json.dumps(status.__dict__, indent=2))
    announce_demo(config, status, open_browser=not getattr(args, "no_open", False))
    return 0 if status.healthy else 1


def _cmd_restart(args: argparse.Namespace) -> int:
    stopped = stop_gateway()
    if stopped.message == "still running after SIGKILL":
        print(json.dumps(stopped.__dict__, indent=2))
        return 1
    return _cmd_start(args)


def _cmd_status(args: argparse.Namespace) -> int:
    _set_config_env(args.config)
    config = load_config(Path(args.config) if args.config else None, strict_env=False)
    status = gateway_status(config)
    print(json.dumps(status.__dict__, indent=2))
    url = dashboard_url(config)
    if url:
        print(f"Dashboard: {url}")
    return 0 if status.healthy else 1


def _cmd_stop(_: argparse.Namespace) -> int:
    status = stop_gateway()
    print(json.dumps(status.__dict__, indent=2))
    return 0 if status.message in {"stopped", "killed", "stale pid file removed", "no pid file"} else 1


def _set_config_env(config: str | None) -> None:
    if config:
        os.environ["EMISSARY_ROUTER_CONFIG"] = config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="er")
    sub = parser.add_subparsers(dest="command", required=True)

    config_path = sub.add_parser("config-path", help="Print the default config path")
    config_path.set_defaults(func=_cmd_config_path)

    init = sub.add_parser("init", help="Create config and set/update API keys")
    init.add_argument("--no-prompt", action="store_true", help="Write a template .env without prompting")
    init.set_defaults(func=_cmd_init)

    validate = sub.add_parser("validate-config", help="Validate config and required env keys")
    validate.add_argument("--config", default=None)
    validate.set_defaults(func=_cmd_validate_config)

    models = sub.add_parser("models", help="List built-in models and config toggles")
    models.add_argument("--config", default=None)
    models.set_defaults(func=_cmd_models)

    start = sub.add_parser("start", help="Start the local gateway in the background")
    start.add_argument("--config", default=None)
    start.add_argument("--no-open", action="store_true", help="Do not open the dashboard in a browser")
    start.set_defaults(func=_cmd_start)

    restart = sub.add_parser("restart", help="Restart the background gateway")
    restart.add_argument("--config", default=None)
    restart.add_argument("--no-open", action="store_true", help="Do not open the dashboard in a browser")
    restart.set_defaults(func=_cmd_restart)

    demo = sub.add_parser("demo", help="Start the gateway and open the split-screen demo")
    demo.add_argument("--config", default=None)
    demo.add_argument("--no-open", action="store_true", help="Do not open the demo in a browser")
    demo.set_defaults(func=_cmd_demo)

    debug = sub.add_parser("debug", help="Run the gateway in the foreground for debugging")
    debug.add_argument("--config", default=None)
    debug.set_defaults(func=_cmd_debug)

    code = sub.add_parser("code", help="Launch Claude Code through Emissary Router")
    code.add_argument("--config", default=None)
    code.add_argument("--claude-command", default=os.environ.get("CLAUDE_COMMAND", "claude"))
    code.add_argument("--dry-run", action="store_true", help="Print launch command and env without exec")
    code.add_argument("--no-open", action="store_true", help="Do not open the dashboard in a browser")
    code.add_argument("claude_args", nargs=argparse.REMAINDER)
    code.set_defaults(func=_cmd_code)

    status = sub.add_parser("status", help="Show Emissary Router gateway status")
    status.add_argument("--config", default=None)
    status.set_defaults(func=_cmd_status)

    stop = sub.add_parser("stop", help="Stop the background gateway")
    stop.set_defaults(func=_cmd_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"er: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
