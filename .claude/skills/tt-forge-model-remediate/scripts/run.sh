#!/bin/bash

if [ -z "$TT_XLA_ROOT" ]; then
  echo "ERROR: TT_XLA_ROOT not set. Abort"
  exit 1
fi

if [ ! -f $TT_XLA_ROOT/.env ]; then
  echo "Env to set HF_TOKEN or TT_COMPILE_ONLY_SYSTEM_DESC doesn't exist"
  exit 1
fi

export XDG_CACHE_HOME="/tmp/tt-forge-cache"
export TT_METAL_CACHE="/tmp/tt-forge-cache"
export HF_HOME="/tmp/tt-forge-cache/huggingface"
export TTMLIR_VENV_DIR="/tmp/tt-forge-venv"
export TT_FORGE_MODELS_ROOT="/data/hf-bringup/tt-xla/third_party/tt_forge_models/worktrees/ip-172-31-30-236-tt-xla-dev+ubuntu+2026-04-23_11-46+hf-bringup-24"

source $TT_XLA_ROOT/.env

cd $TT_XLA_ROOT
source venv/activate

#if [[ -n "$TT_COMPILE_ONLY_SYSTEM_DESC" ]]; then
#  export TT_RANDOM_WEIGHTS=1
#fi
export TTXLA_LOGGER_LEVEL=DEBUG

TEST_NAME="$@"
set -x
pytest "${TEST_NAME}" -svv
