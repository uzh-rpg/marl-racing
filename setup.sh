#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
agi="$(cd -- "${1:?Usage: bash setup.sh /path/to/agilicious_internal}" && pwd)"
python="${PYTHON:-python}"
deps="${RACING_DEPS:-$root/.deps}"
build="${RACING_BUILD:-$root/build}"
mkdir -p "$deps"

checkout() {
  local url="$1" dest="$2" commit="$3"
  if [[ ! -d "$dest/.git" ]]; then
    git clone "$url" "$dest"
    git -C "$dest" checkout "$commit"
  fi
  [[ "$(git -C "$dest" rev-parse HEAD)" == "$commit" ]]
}

"$python" -m pip install -r "$root/requirements.txt"
checkout https://github.com/DLR-RM/stable-baselines3.git "$deps/stable-baselines3" 0532a5719c2bb46fd96b61a7e03dd8cb180c00fc
checkout https://github.com/Stable-Baselines-Team/stable-baselines3-contrib.git "$deps/sb3-contrib" 9cf8b5076f862313e3a1db069b67cba768fa69d6
if ! git -C "$deps/sb3-contrib" apply --reverse --check "$root/patches/sb3_contrib.patch" 2>/dev/null; then
  git -C "$deps/sb3-contrib" apply "$root/patches/sb3_contrib.patch"
fi
if ! git -C "$agi" apply --reverse --check "$root/patches/agilicious.patch" 2>/dev/null; then
  git -C "$agi" apply --check "$root/patches/agilicious.patch"
  git -C "$agi" apply "$root/patches/agilicious.patch"
fi
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DAGILICIOUS_DIR="$agi" -Dpybind11_DIR="$("$python" -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$(command -v "$python")" -DEIGEN_INCLUDE_DIR="${EIGEN_INCLUDE_DIR:-}"
cmake --build "$build" -j "${JOBS:-4}"
