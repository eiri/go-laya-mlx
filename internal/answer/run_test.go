package answer

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/eiri/go-laya-mlx/internal/checkpoint"
	"github.com/eiri/go-laya-mlx/internal/mlx"
	"github.com/eiri/go-laya-mlx/internal/prompt"
)

func TestCachedDecision(t *testing.T) {
	dir, err := checkpoint.Cached()
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("aac6fef/laya-mlx is not cached")
	}
	if err != nil {
		t.Fatal(err)
	}
	model, err := mlx.LoadWeights()
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	tok, err := prompt.NewTokenizer(filepath.Join(dir, "tokenizer"))
	if err != nil {
		t.Fatal(err)
	}
	q := prompt.Question{Type: "choice", Instructions: "Which team?", Options: []prompt.Option{{Label: "billing", Description: "refunds"}, {Label: "tech", Description: "bugs"}}}
	got, seq, err := Run(model, tok, "I was billed twice.", q)
	if err != nil {
		t.Fatal(err)
	}
	if len(seq.Markers) != 2 || got.Choice != "billing" || math.Abs(got.AnswerConfidence-0.9203) > 0.002 || got.CalibrationClamped {
		t.Fatalf("unexpected decision: %+v, sequence: %+v", got, seq)
	}
}
