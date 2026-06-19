#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install --upgrade pip
python -m pip install -e .

mkdir -p "${HOME}/.config/router"
if [ ! -f "${HOME}/.config/router/config.yaml" ]; then
  cp config.example.yaml "${HOME}/.config/router/config.yaml"
fi
if [ ! -f "${HOME}/.config/router/pricing.yaml" ]; then
  cp pricing.example.yaml "${HOME}/.config/router/pricing.yaml"
fi

cat <<'EOF'
router installed.

Next:
  edit ~/.config/router/config.yaml
  edit ~/.config/router/pricing.yaml
  router validate-config
  router start
  router code -- [claude args]
EOF
