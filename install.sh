#!/usr/bin/env bash
set -euo pipefail

# Dev / from-source install. For a normal install just:
#   pip install emissary-router      (or: uv pip install emissary-router)
# Config and keys are created by `er init`, not copied here, so the same flow
# works whether installed from a clone or from a wheel.

cd "$(dirname "$0")"

python -m pip install --upgrade pip
python -m pip install -e .

cat <<'EOF'
Emissary Router installed.

Next:
  er init                 # create config and set your API keys
  er code -- [claude args]

If you previously installed the old local "router" package, remove it:
  python -m pip uninstall router
EOF
