#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_dir="$root/.build/mlx-c"
prefix=${MLX_PREFIX:-"$root/.build/mlx"}
revision=a341b4925024b88b2c593468f16e12f5e17315da
mlx_revision=1f8e74e3f12f31365464a6867c6579f0e9b29d85

# Keep native sources and Metal's module cache out of the repository and user cache.
mkdir -p "$root/.build/home"
if [ ! -d "$source_dir/.git" ]; then
  git clone --depth 1 --branch v0.7.0 https://github.com/ml-explore/mlx-c.git "$source_dir"
fi
[ "$(git -C "$source_dir" rev-parse HEAD)" = "$revision" ] || {
  echo "MLX C revision mismatch" >&2
  exit 1
}

HOME="$root/.build/home" cmake -S "$source_dir" -B "$source_dir/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$prefix"
[ "$(git -C "$source_dir/build/_deps/mlx-src" rev-parse HEAD)" = "$mlx_revision" ] || {
  echo "MLX revision mismatch" >&2
  exit 1
}

HOME="$root/.build/home" cmake --build "$source_dir/build" -j 6
cmake --install "$source_dir/build"
