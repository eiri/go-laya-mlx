package mlx

import (
	"errors"
	"math"
	"os"
	"testing"

	"github.com/eiri/go-laya-mlx/internal/checkpoint"
)

func TestForward(t *testing.T) {
	if _, err := checkpoint.Cached(); errors.Is(err, os.ErrNotExist) {
		t.Skip("aac6fef/laya-mlx is not cached")
	} else if err != nil {
		t.Fatal(err)
	}
	w, err := LoadWeights()
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	for _, tc := range []struct {
		name           string
		ids            []int
		markers        []int
		qtype          int
		logits, action []float32
	}{
		{"choice", []int{50281, 22122, 1953, 27, 6758, 2285, 32, 50282, 50284, 33484, 27, 1275, 41748, 50284, 13817, 27, 19775, 50282, 42, 369, 47045, 7019, 15, 50282}, []int{8, 13}, 0, []float32{4.89453125, 0.23071289}, []float32{4268, -3488}},
		{"score", []int{50281, 18891, 1953, 27, 1359, 21007, 32, 50282, 50284, 1268, 470, 27, 1996, 50284, 1268, 337, 27, 1024, 50282, 42, 369, 47045, 7019, 15, 50282}, []int{8, 13}, 1, []float32{-1.9765625, 3.33203125}, []float32{3148, -2598}},
		{"noul", []int{50281, 79, 3941, 1953, 27, 9876, 253, 7731, 1642, 323, 247, 23005, 32, 50282, 50284, 3221, 27, 642, 13, 253, 3908, 1057, 417, 2186, 50284, 2032, 27, 4754, 13, 253, 3908, 6556, 50282, 42, 369, 47045, 7019, 15, 50282}, []int{14, 24}, 2, []float32{-0.35498047, 0.77783203}, []float32{4708, -3852}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logits, action, err := w.Forward(tc.ids, tc.markers, tc.qtype)
			if err != nil {
				t.Fatal(err)
			}
			if len(logits) != len(tc.logits) || len(action) != len(tc.action) {
				t.Fatalf("output lengths: %d, %d", len(logits), len(action))
			}
			for i, want := range tc.logits {
				if math.Abs(float64(logits[i]-want)) > 0.02 {
					t.Errorf("logit %d: got %g, want %g", i, logits[i], want)
				}
			}
			for i, want := range tc.action {
				if math.Abs(float64(action[i]-want)) > 8 {
					t.Errorf("action %d: got %g, want %g", i, action[i], want)
				}
			}
		})
	}
	if _, _, err := w.Forward([]int{50281}, []int{-1}, 0); err == nil {
		t.Fatal("accepted invalid marker")
	}
}

func TestActionFeatures(t *testing.T) {
	got := actionFeatures([]float32{5})
	want := [4]float32{1, 1, 0, 2.0 / 255}
	for i, value := range want {
		if math.Abs(float64(got[i]-value)) > 1e-6 {
			t.Fatalf("feature %d: got %g, want %g", i, got[i], value)
		}
	}
}
