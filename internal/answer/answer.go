package answer

import (
	"fmt"
	"math"
	"strconv"

	"github.com/eiri/go-laya-mlx/internal/checkpoint"
	"github.com/eiri/go-laya-mlx/internal/prompt"
)

// Answer is one calibrated typed decision.
type Answer struct {
	Type               string             `json:"type"`
	Confidence         float64            `json:"confidence"`
	AnswerConfidence   float64            `json:"answer_confidence"`
	Action             Action             `json:"action"`
	Choice             string             `json:"choice,omitempty"`
	Score              *float64           `json:"score,omitempty"`
	Noul               *float64           `json:"noul,omitempty"`
	Probabilities      map[string]float64 `json:"probabilities,omitempty"`
	Legend             map[string]any     `json:"legend,omitempty"`
	CalibrationClamped bool               `json:"calibration_clamped,omitempty"`
}

type Action struct {
	ActProbability float64 `json:"act_probability"`
}

// Format applies checkpoint temperatures to raw scores and selects an answer.
func Format(q prompt.Question, logits, action []float32, cfg checkpoint.Agent) (Answer, error) {
	kind := map[string]int{"choice": 0, "score": 1, "noul": 2}
	qt, ok := kind[q.Type]
	if !ok {
		return Answer{}, fmt.Errorf("unknown question type %q", q.Type)
	}
	k := len(logits)
	if k == 0 || len(action) != 2 {
		return Answer{}, fmt.Errorf("invalid model output length")
	}
	if q.Type == "choice" && len(q.Options) != k || q.Type == "score" && len(q.Levels) != k || q.Type == "noul" && k != 2 {
		return Answer{}, fmt.Errorf("logit count does not match question")
	}

	// Per-option temperatures override the question-type fallback.
	size := "11+"
	if k <= 2 {
		size = "2"
	} else if k <= 5 {
		size = "3-5"
	} else if k <= 10 {
		size = "6-10"
	}
	temp := float64(1)
	if len(cfg.Temperature) == 3 {
		temp = cfg.Temperature[qt]
	}
	if override, ok := cfg.TemperatureByOpt[q.Type+":"+size]; ok {
		temp = override
	}
	if !finite(temp) || temp <= 0 {
		return Answer{}, fmt.Errorf("invalid calibration temperature")
	}
	clamped := max(0.5, min(5, temp))
	values := make([]float64, k)
	for i, value := range logits {
		if !finite(float64(value)) {
			return Answer{}, fmt.Errorf("non-finite option score")
		}
		values[i] = float64(value) / clamped
	}
	p, err := softmax(values)
	if err != nil {
		return Answer{}, err
	}
	if !finite(float64(action[0])) || !finite(float64(action[1])) {
		return Answer{}, fmt.Errorf("non-finite action score")
	}
	act, err := softmax([]float64{float64(action[0]), float64(action[1])})
	if err != nil {
		return Answer{}, err
	}

	best := 0
	entropy := float64(0)
	for i, value := range p {
		if value > p[best] {
			best = i
		}
		entropy -= value * math.Log(max(value, 1e-12))
	}
	confidence := float64(1)
	if k > 1 {
		confidence = max(0, min(1, 1-entropy/math.Log(float64(k))))
	}
	out := Answer{Type: q.Type, Confidence: rounded(confidence), AnswerConfidence: rounded(p[best]),
		Action: Action{ActProbability: rounded(act[0])}, CalibrationClamped: clamped != temp}

	switch q.Type {
	case "choice":
		out.Choice = q.Options[best].Label
		out.Probabilities = make(map[string]float64, k)
		for i, opt := range q.Options {
			if opt.Label == "" {
				return Answer{}, fmt.Errorf("empty choice label")
			}
			if _, exists := out.Probabilities[opt.Label]; exists {
				return Answer{}, fmt.Errorf("duplicate choice label %q", opt.Label)
			}
			out.Probabilities[opt.Label] = rounded(p[i])
		}
	case "score":
		score := float64(0)
		out.Legend = make(map[string]any, k)
		out.Probabilities = make(map[string]float64, k)
		for i, level := range q.Levels {
			score += float64(i) * p[i]
			key := strconv.Itoa(i)
			out.Legend[key] = level
			out.Probabilities[key] = rounded(p[i])
		}
		score = rounded(score)
		out.Score = &score
	case "noul":
		yes := rounded(p[1])
		out.Noul = &yes
		out.Confidence = rounded(max(p[1], 1-p[1]))
	}
	return out, nil
}

func softmax(values []float64) ([]float64, error) {
	peak := values[0]
	for _, v := range values {
		peak = max(peak, v)
	}
	result := make([]float64, len(values))
	total := float64(0)
	for i, v := range values {
		result[i] = math.Exp(v - peak)
		total += result[i]
	}
	if !finite(total) || total == 0 {
		return nil, fmt.Errorf("invalid model scores")
	}
	for i := range result {
		result[i] /= total
	}
	return result, nil
}

func finite(n float64) bool     { return !math.IsNaN(n) && !math.IsInf(n, 0) }
func rounded(n float64) float64 { return math.RoundToEven(n*1e4) / 1e4 }
