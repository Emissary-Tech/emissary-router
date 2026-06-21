from __future__ import annotations

import hashlib
from pathlib import Path


CLASSIFIER_INPUT_SHA256 = "cb0dbc5ae66a77a140a0b2752fe9d1977c2163b6e13e9c39e58bd13f481cb271"


def test_classifier_input_contract_has_not_drifted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    classifier_input = repo_root / "src" / "router" / "routing" / "classifier_input.py"

    assert hashlib.sha256(classifier_input.read_bytes()).hexdigest() == CLASSIFIER_INPUT_SHA256
