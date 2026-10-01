package laya_test

import (
	"errors"
	"math"
	"os"
	"reflect"
	"strings"
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

// Compare public predictions with Laya-MLX 0.3.0, MLX 0.32.2, FP16.
func TestReferenceCases(t *testing.T) {
	model, err := laya.Load()
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("aac6fef/laya-mlx is not cached")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	team := laya.Question{Type: "choice", Instructions: "Which team?", Options: []laya.Option{
		{Label: "billing", Description: "refunds"}, {Label: "tech", Description: "bugs"},
	}}
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	conversation := []message{
		{Role: "user", Content: strings.Repeat("hello ", 600)},
		{Role: "user", Content: "I was billed twice."},
	}
	// Struct fields retain the Python fixture's JSON key order; Go maps do not.
	type invoice struct {
		ID        int  `json:"id"`
		Duplicate bool `json:"duplicate"`
	}
	structured := struct {
		Account struct {
			Tier     string    `json:"tier"`
			Invoices []invoice `json:"invoices"`
		} `json:"account"`
		Request string `json:"request"`
	}{Request: "I was billed twice."}
	structured.Account.Tier = "basic"
	structured.Account.Invoices = []invoice{{ID: 3, Duplicate: true}}

	for _, tc := range []struct {
		name, choice            string
		state                   any
		question                laya.Question
		probability, confidence float64
		input, total, dropped   int
	}{
		{"one option", "only", strings.Repeat("hello ", 1000), laya.Question{Type: "choice", Instructions: "choose", Options: []laya.Option{{Label: "only"}}}, 1, 1, 512, 1001, 499},
		{"long state", "billing", strings.Repeat("I was billed twice. ", 100), team, 0.8034, 0.2849, 512, 501, 8},
		{"conversation", "tech", conversation, team, 0.5856, 0.0213, 512, 629, 136},
		{"structured", "", structured, laya.Question{Type: "noul", Instructions: "Does the customer ask for money back?"}, 0.7828, 0.7828, 71, 37, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A second call also checks that inference does not change its answer.
			var first laya.Answer
			for attempt := range 2 {
				got, usage, err := model.Predict(tc.state, tc.question)
				if err != nil {
					t.Fatal(err)
				}
				p := got.AnswerConfidence
				if tc.name == "structured" {
					p = *got.Noul
				}
				if got.Choice != tc.choice || math.Abs(p-tc.probability) > 0.002 ||
					math.Abs(got.Confidence-tc.confidence) > 0.002 || got.Action.ActProbability != 1 ||
					usage != (laya.Usage{InputTokens: tc.input, StateTokens: tc.total, StateDropped: tc.dropped}) {
					t.Fatalf("answer: %+v, usage: %+v", got, usage)
				}
				if attempt > 0 && !reflect.DeepEqual(got, first) {
					t.Fatalf("answer changed: %+v, then %+v", first, got)
				}
				first = got
			}
		})
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
