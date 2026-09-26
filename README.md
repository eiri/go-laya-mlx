# Laya-MLX in Go flavour

Go inference for Laya on Apple Silicon.

## Build

Install Xcode and CMake on your Mac. MLX runs through a native C library, so the Go build uses cgo.

```sh
mise install
mise run native
mise run check
```

`mise run native` builds pinned MLX C v0.7.0 and MLX v0.32.2 under `.build/`. It needs network access on the first run. `mise run check` runs a Metal operation from Go. If you run Go commands without mise, set `CGO_ENABLED=1`. Programs that use MLX need the libraries in `.build/mlx/lib` at runtime.

To test checkpoint loading and the first MLX embedding operation, cache `aac6fef/laya-mlx` with Hugging Face, then run `mise run test`. The library reads `HF_HUB_CACHE` when set. Otherwise, it uses `HF_HOME`, `XDG_CACHE_HOME`, or the default cache directory. Tests skip model checks when the model is not cached.
