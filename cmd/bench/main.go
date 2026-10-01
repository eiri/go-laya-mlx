package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"time"

	laya "github.com/eiri/go-laya-mlx"
	"github.com/eiri/go-laya-mlx/internal/mlx"
)

type result struct {
	Name        string    `json:"name"`
	Questions   int       `json:"questions"`
	InputTokens int       `json:"input_tokens"`
	Warmups     int       `json:"warmups"`
	SamplesMS   []float64 `json:"samples_ms"`
	P50MS       float64   `json:"p50_ms"`
	P95MS       float64   `json:"p95_ms"`
	MeanMS      float64   `json:"mean_ms"`
	PerSecond   float64   `json:"questions_per_second"`
	ActiveBytes uint64    `json:"mlx_active_bytes"`
	PeakBytes   uint64    `json:"mlx_peak_bytes"`
}

type workload struct {
	name      string
	state     any
	question  laya.Question
	questions int
	samples   int
}

func percentile(sorted []float64, q float64) float64 {
	at := q * float64(len(sorted)-1)
	lower := int(at)
	upper := min(lower+1, len(sorted)-1)
	return sorted[lower] + (sorted[upper]-sorted[lower])*(at-float64(lower))
}

func measure(model *laya.Model, work workload) (result, error) {
	const warmups = 20
	out := result{Name: work.name, Questions: work.questions, Warmups: warmups}
	call := func() (int, error) {
		count := 0
		for range work.questions {
			answer, usage, err := model.Predict(work.state, work.question)
			if err != nil {
				return 0, err
			}
			if answer.Type != work.question.Type || usage.InputTokens == 0 {
				return 0, fmt.Errorf("invalid prediction")
			}
			count += usage.InputTokens
		}
		return count, nil
	}

	// Warm up Metal, then measure complete Predict calls without model loading.
	for range warmups {
		if _, err := call(); err != nil {
			return out, err
		}
	}
	if err := mlx.ResetPeak(); err != nil {
		return out, err
	}
	out.SamplesMS = make([]float64, work.samples)
	for i := range out.SamplesMS {
		start := time.Now()
		count, err := call()
		if err != nil {
			return out, err
		}
		out.SamplesMS[i] = float64(time.Since(start)) / float64(time.Millisecond)
		out.InputTokens = count
		out.MeanMS += out.SamplesMS[i]
	}
	out.MeanMS /= float64(work.samples)
	out.PerSecond = float64(work.questions) * 1000 / out.MeanMS
	var err error
	out.ActiveBytes, out.PeakBytes, err = mlx.Memory()
	if err != nil {
		return out, err
	}
	sorted := slices.Clone(out.SamplesMS)
	slices.Sort(sorted)
	out.P50MS = percentile(sorted, 0.5)
	out.P95MS = percentile(sorted, 0.95)
	return out, nil
}

func run() error {
	model, err := laya.Load()
	if err != nil {
		return err
	}
	defer model.Close()
	q := laya.Question{Type: "choice", Instructions: "Which team?", Options: []laya.Option{
		{Label: "billing", Description: "refunds"}, {Label: "tech", Description: "bugs"},
	}}
	works := []workload{
		{"short", "I was billed twice.", q, 1, 100},
		{"50 serial", "I was billed twice.", q, 50, 30},
		{"long", strings.Repeat("I was billed twice. ", 100), q, 1, 40},
	}
	report := struct {
		When    string   `json:"utc"`
		Go      string   `json:"go"`
		Results []result `json:"results"`
	}{When: time.Now().UTC().Format(time.RFC3339), Go: runtime.Version()}
	for _, work := range works {
		row, err := measure(model, work)
		if err != nil {
			return fmt.Errorf("%s: %w", work.name, err)
		}
		report.Results = append(report.Results, row)
	}
	return json.NewEncoder(os.Stdout).Encode(report)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
