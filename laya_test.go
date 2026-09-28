package laya_test

import (
	"errors"
	"math"
	"os"
	"testing"

	laya "github.com/eiri/go-laya-mlx"
)

func TestModel(t *testing.T) {
	model, err := laya.Load()
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("aac6fef/laya-mlx is not cached")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	q := laya.Question{Type: "choice", Instructions: "Which team?", Options: []laya.Option{
		{Label: "billing", Description: "refunds"},
		{Label: "tech", Description: "bugs"},
	}}
	for range 2 {
		got, usage, err := model.Predict("I was billed twice.", q)
		if err != nil {
			t.Fatal(err)
		}
		if got.Choice != "billing" || math.Abs(got.AnswerConfidence-0.9203) > 0.002 {
			t.Fatalf("unexpected answer: %+v", got)
		}
		if usage.InputTokens <= usage.StateTokens || usage.StateTokens == 0 || usage.StateDropped != 0 {
			t.Fatalf("unexpected usage: %+v", usage)
		}
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := model.Predict("state", q); err == nil {
		t.Fatal("prediction after close succeeded")
	}
}

func TestNilModel(t *testing.T) {
	var model *laya.Model
	if _, _, err := model.Predict("state", laya.Question{}); err == nil {
		t.Fatal("nil model prediction succeeded")
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
}
