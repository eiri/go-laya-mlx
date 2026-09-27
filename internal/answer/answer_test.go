package answer

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/eiri/go-laya-mlx/internal/checkpoint"
	"github.com/eiri/go-laya-mlx/internal/prompt"
)

func TestFormat(t *testing.T) {
	cfg := checkpoint.Agent{
		Temperature:      []float64{1.6369030475616455, 1.2514300346374512, 1.983399510383606},
		TemperatureByOpt: map[string]float64{"choice:2": 1.9063563346862793},
	}
	for _, tc := range []struct {
		name       string
		q          prompt.Question
		logits     []float32
		prob       float64
		confidence float64
	}{
		{"choice", prompt.Question{Type: "choice", Options: []prompt.Option{{Label: "billing"}, {Label: "tech"}}}, []float32{4.89453125, 0.230712890625}, 0.9203, 0.5989},
		{"score", prompt.Question{Type: "score", Levels: []any{"later", "now"}}, []float32{-1.9765625, 3.33203125}, 0.9858, 0.8927},
		{"noul", prompt.Question{Type: "noul"}, []float32{-0.35498046875, 0.77783203125}, 0.639, 0.639},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := Format(tc.q, tc.logits, []float32{4268, -3488}, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if out.AnswerConfidence != tc.prob || out.Confidence != tc.confidence || out.Action.ActProbability != 1 {
				t.Fatalf("got %+v", out)
			}
			switch tc.name {
			case "choice":
				if out.Choice != "billing" || out.Probabilities["tech"] != 0.0797 {
					t.Fatalf("got %+v", out)
				}
			case "score":
				if out.Score == nil || *out.Score != 0.9858 || out.Legend["0"] != "later" {
					t.Fatalf("got %+v", out)
				}
			case "noul":
				if out.Noul == nil || *out.Noul != 0.639 || out.Probabilities != nil {
					t.Fatalf("got %+v", out)
				}
			}
		})
	}
}

func TestClamped(t *testing.T) {
	cfg := checkpoint.Agent{TemperatureByOpt: map[string]float64{"choice:11+": 0.1}}
	q := prompt.Question{Type: "choice"}
	logits := make([]float32, 11)
	for i := range logits {
		q.Options = append(q.Options, prompt.Option{Label: string(rune('a' + i))})
	}
	logits[0] = 1
	out, err := Format(q, logits, []float32{0, 0}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !out.CalibrationClamped || out.Action.ActProbability != 0.5 {
		t.Fatalf("got %+v", out)
	}
	if out.AnswerConfidence < 0.424 || out.AnswerConfidence > 0.425 {
		t.Fatalf("unexpected clamped probability: %g", out.AnswerConfidence)
	}
}

func TestInvalid(t *testing.T) {
	q := prompt.Question{Type: "choice", Options: []prompt.Option{{Label: "a"}, {Label: "b"}}}
	for _, tc := range []struct {
		q              prompt.Question
		logits, action []float32
		cfg            checkpoint.Agent
	}{
		{q, []float32{1}, []float32{0, 0}, checkpoint.Agent{}},
		{q, []float32{1, 2}, []float32{0}, checkpoint.Agent{}},
		{q, []float32{float32(math.NaN()), 1}, []float32{0, 0}, checkpoint.Agent{}},
		{q, []float32{1, 2}, []float32{0, 0}, checkpoint.Agent{Temperature: []float64{0, 1, 1}}},
	} {
		if _, err := Format(tc.q, tc.logits, tc.action, tc.cfg); err == nil {
			t.Fatalf("accepted invalid input: %+v", tc)
		}
	}
}

func TestZeroScoreJSON(t *testing.T) {
	q := prompt.Question{Type: "score", Levels: []any{"only"}}
	out, err := Format(q, []float32{2}, []float32{0, 0}, checkpoint.Agent{})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"score":0`) {
		t.Fatalf("score missing from %s", data)
	}
}
