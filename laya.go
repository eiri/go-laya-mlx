// Package laya runs typed Laya decisions with a locally cached MLX checkpoint.
package laya

import (
	"fmt"
	"path/filepath"
	"sync"

	"github.com/eiri/go-laya-mlx/internal/answer"
	"github.com/eiri/go-laya-mlx/internal/checkpoint"
	"github.com/eiri/go-laya-mlx/internal/mlx"
	"github.com/eiri/go-laya-mlx/internal/prompt"
)

type Option = prompt.Option
type Question = prompt.Question
type Answer = answer.Answer
type Action = answer.Action

// Usage counts input tokens and state tokens removed to fit the model.
type Usage struct {
	InputTokens  int
	StateTokens  int
	StateDropped int
}

// Model owns the checkpoint weights and tokenizer.
type Model struct {
	mu      sync.RWMutex
	weights *mlx.Weights
	tok     *prompt.Tokenizer
}

// Load opens the cached aac6fef/laya-mlx checkpoint.
func Load() (*Model, error) {
	dir, err := checkpoint.Cached()
	if err != nil {
		return nil, err
	}
	tok, err := prompt.NewTokenizer(filepath.Join(dir, "tokenizer"))
	if err != nil {
		return nil, err
	}
	weights, err := mlx.LoadWeights()
	if err != nil {
		return nil, err
	}
	return &Model{weights: weights, tok: tok}, nil
}

// Predict evaluates one question. Calls on the same model are safe concurrently.
func (m *Model) Predict(state any, q Question) (Answer, Usage, error) {
	if m == nil {
		return Answer{}, Usage{}, fmt.Errorf("model is nil")
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.weights == nil {
		return Answer{}, Usage{}, fmt.Errorf("model is closed")
	}
	result, seq, err := answer.Run(m.weights, m.tok, state, q)
	if err != nil {
		return Answer{}, Usage{}, err
	}
	return result, Usage{InputTokens: len(seq.IDs), StateTokens: seq.StateTokens, StateDropped: seq.StateDropped}, nil
}

// Close releases the model. Repeated calls are safe.
func (m *Model) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.weights == nil {
		return nil
	}
	err := m.weights.Close()
	m.weights = nil
	return err
}
