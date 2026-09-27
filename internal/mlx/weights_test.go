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

	encoded, err := w.Encode([]int{50281, 12092})
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) != len(emb) {
		t.Fatalf("encoder size: %d", len(encoded))
	}
	for i, want := range []float32{-0.08642578, 0.00811768, 0.11743164, 0.01523590} {
		if diff := encoded[i] - want; diff < -0.002 || diff > 0.002 {
			t.Fatalf("encoder[%d] = %g, want %g", i, encoded[i], want)
		}
	}

	long := make([]int, 130)
	long[0] = 50281
	for i := 1; i < len(long); i++ {
		long[i] = 12092
	}
	encoded, err = w.Encode(long)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{-0.1553955, 0.30688477, -1.40722656, -0.87060547} {
		if diff := encoded[i] - want; diff < -0.03 || diff > 0.03 {
			t.Fatalf("long encoder[%d] = %g, want %g", i, encoded[i], want)
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
