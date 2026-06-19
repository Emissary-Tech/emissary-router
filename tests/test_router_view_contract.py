from __future__ import annotations

import hashlib
from pathlib import Path


ROUTER_VIEW_SHA256 = "9b461c10ec9ae06121013adf046cd82a70c26b37e7d2a631ac32590867b9c9a0"


def test_router_view_contract_has_not_drifted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    view = repo_root / "src" / "router" / "routing" / "view.py"

    assert hashlib.sha256(view.read_bytes()).hexdigest() == ROUTER_VIEW_SHA256
