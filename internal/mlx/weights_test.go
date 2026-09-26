package mlx

import (
	"errors"
	"os"
	"sync"
	"testing"

	"github.com/eiri/go-laya-mlx/internal/checkpoint"
)

func TestWeights(t *testing.T) {
	_, err := checkpoint.Cached()
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("aac6fef/laya-mlx is not cached")
	}
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadWeights()
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	values, err := w.Tensor("temperature")
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 3 || values[0] != 1 || values[1] != 1 || values[2] != 1 {
		t.Fatalf("unexpected checkpoint temperatures: %v", values)
	}
	emb, err := w.Embed([]int{50281, 12092})
	if err != nil {
		t.Fatal(err)
	}
	if len(emb) != 2*w.Config.HiddenSize {
		t.Fatalf("embedding size: %d", len(emb))
	}
	// The first values match MLX Python FP16 output for the same checkpoint.
	want := []float32{0.21435547, 0.18103027, 0.22961426, -0.04309082}
	for i, value := range want {
		if diff := emb[i] - value; diff < -1e-4 || diff > 1e-4 {
			t.Fatalf("embedding[%d] = %g, want %g", i, emb[i], value)
		}
	}

	// Calls from other goroutines must not reuse a thread-affine MLX stream.
	var group sync.WaitGroup
	for range 4 {
		group.Add(1)
		go func() {
			defer group.Done()
			if _, err := w.Embed([]int{50281}); err != nil {
				t.Error(err)
			}
		}()
	}
	group.Wait()

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Tensor("temperature"); err == nil {
		t.Fatal("read closed weights")
	}
}
