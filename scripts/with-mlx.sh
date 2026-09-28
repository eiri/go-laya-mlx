#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
prefix=${MLX_PREFIX:-"$root/.build/mlx"}
case "$prefix" in
  /*) ;;
  *) echo "MLX_PREFIX must be an absolute path" >&2; exit 1 ;;
esac
case "$prefix" in
  *' '*) echo "MLX_PREFIX cannot contain spaces" >&2; exit 1 ;;
esac

if [ ! -f "$prefix/include/mlx/c/mlx.h" ] || [ ! -f "$prefix/lib/libmlxc.dylib" ] || [ ! -f "$prefix/lib/libmlx.dylib" ] || [ ! -f "$prefix/lib/mlx.metallib" ]; then
  echo "MLX C headers, libraries, or Metal shaders missing in $prefix" >&2
  exit 1
fi

# Pass the installation paths to cgo and embed the runtime library path.
export CGO_ENABLED=1
export CGO_CFLAGS="-I$prefix/include ${CGO_CFLAGS:-}"
export CGO_LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib ${CGO_LDFLAGS:-}"
exec "$@"
