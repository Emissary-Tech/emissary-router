#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install --upgrade pip
python -m pip install -e .

EMISSARY_ROUTER_HOME="${EMISSARY_ROUTER_HOME:-${HOME}/.emissary-router}"

mkdir -p "${EMISSARY_ROUTER_HOME}"
if [ ! -f "${EMISSARY_ROUTER_HOME}/config.yaml" ]; then
  cp config.example.yaml "${EMISSARY_ROUTER_HOME}/config.yaml"
fi
if [ ! -f "${EMISSARY_ROUTER_HOME}/pricing.yaml" ]; then
  cp pricing.example.yaml "${EMISSARY_ROUTER_HOME}/pricing.yaml"
fi

cat <<EOF
emissary-router installed.

Next:
  edit ${EMISSARY_ROUTER_HOME}/config.yaml
  edit ${EMISSARY_ROUTER_HOME}/pricing.yaml
  emissary-router validate-config
  emissary-router start
  emissary-router code -- [claude args]

If you previously installed the old local "router" package, its command may still
exist until you remove it:
  python -m pip uninstall router
EOF
