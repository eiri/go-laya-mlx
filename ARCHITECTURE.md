# Architecture

This library answers one typed question about a state per call. A typed question asks for a choice, a rubric score, or a probability (`noul`). Inference runs in Go and MLX C on Apple Silicon.

`Load` creates a `Model` from the cached `aac6fef/laya-mlx` checkpoint. `Predict` returns an `Answer` and token `Usage`. `Close` releases the model. The model lock keeps `Close` from interrupting `Predict`. MLX uses a second lock to protect its shared streams.

The data follows this path:

```text
state + Question → prompt tokens and option markers → MLX scores → calibrated Answer + Usage
```

- `internal/checkpoint/` finds the local Hugging Face snapshot. It reads the configuration and checks tensor names, shapes, types, and byte ranges.
- `internal/prompt/` loads `tokenizer.json` and makes one token sequence. It keeps options in the order supplied by the caller. Each option gets a marker that the model uses for scoring. It limits the sequence to the model's token budget and counts dropped state tokens. Conversation slices keep the end of the state when truncated. Go sorts JSON map keys, so maps can produce a different prompt from an ordered Python dictionary that was used for the port.
- `internal/mlx/` loads the weights and runs ModernBERT and the decision layers on Metal through cgo. It uses the markers to return raw option scores and action scores. `Forward` takes one unpadded sequence. MLX memory helpers support the benchmark.
- `internal/answer/` calls prompt preparation and inference, then applies the checkpoint temperatures. It caps temperatures at `[0.5, 5]` and marks affected answers.

`scripts/build-mlx.sh` builds pinned MLX C and MLX. `scripts/with-mlx.sh` supplies the header, library, and runtime paths to cgo. Applications need those libraries and `mlx.metallib` at runtime.

`cmd/bench/` measures calls to the public API; `benchmarks/m2-pro.json` holds the recorded samples.

Tests compare selected outputs with [Laya-MLX](https://github.com/mizorewww/laya-mlx), including long inputs, truncation, and repeated calls.

This library does not batch questions, route between models, or download checkpoints.
