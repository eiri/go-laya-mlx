# Laya-MLX in Go flavour

Go inference for Laya on Apple Silicon.

## Motivation

I needed a classifier to sort out my messy ebooks collection and didn't want to muck with python. Also - hype.

## Build

Install Xcode and CMake. MLX runs through a native C library, so the Go build uses cgo.

```sh
mise install
mise run native
mise run check
```

`mise run native` builds pinned MLX C v0.7.0 and MLX v0.32.2 under `.build/`. The first build needs network access. `mise run check` runs Go tests and builds the library.

## Use from another Go module

Build MLX C from this checkout into an absolute path. Keep the installed libraries and `mlx.metallib` in that path at runtime.

```sh
export MLX_PREFIX="$HOME/.mlx-laya"
mise run native
```

In the module that imports this library, set the cgo paths before building or running Go. The runtime path in `CGO_LDFLAGS` lets the executable find both MLX libraries.

```sh
export CGO_ENABLED=1
export CGO_CFLAGS="-I$MLX_PREFIX/include"
export CGO_LDFLAGS="-L$MLX_PREFIX/lib -Wl,-rpath,$MLX_PREFIX/lib"
go get github.com/eiri/go-laya-mlx
```

Then call the library from Go:

```go
package main

import (
    "log"

    laya "github.com/eiri/go-laya-mlx"
)

func main() {
    model, err := laya.Load()
    if err != nil { log.Fatal(err) }
    defer model.Close()

    answer, usage, err := model.Predict("I was billed twice.", laya.Question{
        Type: "choice", Instructions: "Which team?",
        Options: []laya.Option{{Label: "billing", Description: "refunds"}, {Label: "tech", Description: "bugs"}},
    })
    if err != nil { log.Fatal(err) }
    log.Printf("%s: %+v", answer.Choice, usage)
}
```

`Load` uses only the cached `aac6fef/laya-mlx` model, so `hf download aac6fef/laya-mlx` first. Set `HF_HUB_CACHE` if your cache is outside the default Hugging Face location. The lookup also supports `HF_HOME`and `XDG_CACHE_HOME`. Tests skip model checks when the cache is absent. `Predict` handles one question per call.

## Performance on M2 Pro

| FP16, one Go `Predict` call        |            Result |
| ---------------------------------- | ----------------: |
| Short input, median (P50)          |          59.91 ms |
| Short input, 95th percentile (P95) |          98.57 ms |
| 50 short questions, sequential     | 17.61 questions/s |
| Long input (512 tokens), median    |         188.88 ms |
| Peak MLX allocation, short input   |           905 MiB |

Measured on an M2 Pro with 16 GPU cores and 32 GiB of memory, macOS 15.7.9, Go 1.26.8, MLX C 0.7.0, and MLX 0.32.2. The short input is the billing choice above (24 tokens). Each number includes prompt preparation, tokenization, Metal inference, and answer formatting, but excludes model loading. GPU results finish before `Predict` returns. The short run has 20 warm-up calls and 100 timed calls. The 50-question run repeats 50 separate calls 30 times after 20 warm-up runs. This library does not batch questions, so its throughput is not comparable to upstream's batched throughput. The peak is MLX allocator memory after clearing its cache and resetting its counter, not total process memory. [All samples](benchmarks/m2-pro.json) and the [benchmark program](cmd/bench/main.go) are available. To rerun the test on your machine, use `./scripts/with-mlx.sh go run ./cmd/bench > benchmarks/m2-pro.json`.

## Licence

[MIT](LICENSE), also read [NOTICE](NOTICE)
