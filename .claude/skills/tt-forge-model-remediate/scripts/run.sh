#!/bin/bash

if [ -z "$TT_XLA_ROOT" ]; then
  echo "ERROR: TT_XLA_ROOT not set. Abort"
  exit 1
fi

if [ ! -f $TT_XLA_ROOT/.env ]; then
  echo "Env to set HF_TOKEN or TT_COMPILE_ONLY_SYSTEM_DESC doesn't exist"
  exit 1
fi

WORKTREE_DIR=$PWD
export XDG_CACHE_HOME="$PWD/.cache"
export TT_METAL_CACHE="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
export TTMLIR_VENV_DIR=$PWD/.local_venv

source $TT_XLA_ROOT/.env

# Override TT_FORGE_MODELS_ROOT so the test uses this worktree's model files
# instead of whatever the shared .env may have set (set by other parallel jobs).
export TT_FORGE_MODELS_ROOT=$WORKTREE_DIR

cd $TT_XLA_ROOT
source venv/activate

#if [[ -n "$TT_COMPILE_ONLY_SYSTEM_DESC" ]]; then
#  export TT_RANDOM_WEIGHTS=1
#fi
export TTXLA_LOGGER_LEVEL=DEBUG

TEST_NAME="$@"
set -x
pytest "${TEST_NAME}" -svv
