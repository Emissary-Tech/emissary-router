from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import uvicorn

from emissary_router.config import (
    load_config,
    load_pricing,
    unresolved_env_paths,
    user_config_path,
    user_pricing_path,
)
from emissary_router.launch import exec_claude, gateway_status, stop_gateway
from emissary_router.launch import ensure_gateway


def _cmd_config_path(_: argparse.Namespace) -> int:
    print(user_config_path())
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else user_config_path()
    pricing_path = Path(args.pricing) if args.pricing else user_pricing_path()
    config = load_config(config_path, strict_env=False)
    pricing = load_pricing(pricing_path, strict_env=False)
    missing_prices = [
        model_name
        for model_name in config.models
        if model_name not in pricing.pricing
    ]
    unresolved_env = {
        "config": unresolved_env_paths(config_path),
        "pricing": unresolved_env_paths(pricing_path),
    }
    result = {
        "ok": not missing_prices,
        "models": sorted(config.models),
        "providers": sorted(config.providers),
        "missing_prices": missing_prices,
        "unresolved_env": unresolved_env,
    }
    print(json.dumps(result, indent=2))
    return 0 if not missing_prices else 2


def _cmd_code(args: argparse.Namespace) -> int:
    if args.config:
        os.environ["EMISSARY_ROUTER_CONFIG"] = args.config
    if args.pricing:
        os.environ["EMISSARY_ROUTER_PRICING"] = args.pricing
    config_path = (Path(args.config) if args.config else user_config_path()).expanduser().resolve()
    pricing_path = (Path(args.pricing) if args.pricing else user_pricing_path()).expanduser().resolve()
    config = load_config(config_path)
    load_pricing(pricing_path)
    claude_args = list(args.claude_args)
    if claude_args and claude_args[0] == "--":
        claude_args = claude_args[1:]
    return exec_claude(
        config=config,
        config_path=config_path,
        pricing_path=pricing_path,
        claude_command=args.claude_command,
        claude_args=claude_args,
        dry_run=args.dry_run,
    )


def _cmd_debug(args: argparse.Namespace) -> int:
    if args.config:
        os.environ["EMISSARY_ROUTER_CONFIG"] = args.config
    if args.pricing:
        os.environ["EMISSARY_ROUTER_PRICING"] = args.pricing
    config = load_config(Path(args.config) if args.config else None)
    uvicorn.run(
        "emissary_router.server:create_app",
        host=config.server.host,
        port=config.server.port,
        factory=True,
    )
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    if args.config:
        os.environ["EMISSARY_ROUTER_CONFIG"] = args.config
    if args.pricing:
        os.environ["EMISSARY_ROUTER_PRICING"] = args.pricing
    config_path = (Path(args.config) if args.config else user_config_path()).expanduser().resolve()
    pricing_path = (Path(args.pricing) if args.pricing else user_pricing_path()).expanduser().resolve()
    config = load_config(config_path)
    load_pricing(pricing_path)
    status = ensure_gateway(config, config_path, pricing_path)
    print(json.dumps(status.__dict__, indent=2))
    return 0 if status.healthy else 1


def _cmd_restart(args: argparse.Namespace) -> int:
    stopped = stop_gateway()
    if stopped.message == "still running after SIGTERM":
        print(json.dumps(stopped.__dict__, indent=2))
        return 1
    return _cmd_start(args)


def _cmd_status(args: argparse.Namespace) -> int:
    if args.config:
        os.environ["EMISSARY_ROUTER_CONFIG"] = args.config
    config = load_config(Path(args.config) if args.config else None, strict_env=False)
    status = gateway_status(config)
    print(json.dumps(status.__dict__, indent=2))
    return 0 if status.healthy else 1


def _cmd_stop(_: argparse.Namespace) -> int:
    status = stop_gateway()
    print(json.dumps(status.__dict__, indent=2))
    return 0 if status.message in {"stopped", "stale pid file removed", "no pid file"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emissary-router")
    sub = parser.add_subparsers(dest="command", required=True)

    config_path = sub.add_parser("config-path", help="Print the default config path")
    config_path.set_defaults(func=_cmd_config_path)

    validate = sub.add_parser("validate-config", help="Validate config and pricing files")
    validate.add_argument("--config", default=None)
    validate.add_argument("--pricing", default=None)
    validate.set_defaults(func=_cmd_validate_config)

    start = sub.add_parser("start", help="Start the local gateway in the background")
    start.add_argument("--config", default=None)
    start.add_argument("--pricing", default=None)
    start.set_defaults(func=_cmd_start)

    restart = sub.add_parser("restart", help="Restart the background gateway")
    restart.add_argument("--config", default=None)
    restart.add_argument("--pricing", default=None)
    restart.set_defaults(func=_cmd_restart)

    debug = sub.add_parser("debug", help="Run the gateway in the foreground for debugging")
    debug.add_argument("--config", default=None)
    debug.add_argument("--pricing", default=None)
    debug.set_defaults(func=_cmd_debug)

    code = sub.add_parser("code", help="Launch Claude Code through Emissary Router")
    code.add_argument("--config", default=None)
    code.add_argument("--pricing", default=None)
    code.add_argument("--claude-command", default=os.environ.get("CLAUDE_COMMAND", "claude"))
    code.add_argument("--dry-run", action="store_true", help="Print launch command and env without exec")
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
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
