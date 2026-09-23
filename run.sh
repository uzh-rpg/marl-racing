#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export FLIGHTMARE_PATH="$root"
export RACING_PARAMS="$root/params"
export PYTHONPATH="$root/python:${RACING_BUILD:-$root/build}/python:${RACING_DEPS:-$root/.deps}/stable-baselines3:${RACING_DEPS:-$root/.deps}/sb3-contrib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "${PYTHON:-python}" "$root/python/$1.py" "${@:2}"
