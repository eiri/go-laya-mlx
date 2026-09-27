package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// Forward runs the decision layers on one unpadded sequence.
// qtype is 0 for choice, 1 for score, and 2 for noul.
// It returns raw option logits and raw action logits, before calibration.
func (w *Weights) Forward(ids, markers []int, qtype int) ([]float32, []float32, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if qtype < 0 || qtype > 2 || len(markers) == 0 || len(ids) > w.Agent.MaxLen {
		return nil, nil, fmt.Errorf("invalid decision input")
	}
	for i, pos := range markers {
		if pos < 0 || pos >= len(ids) || i > 0 && pos <= markers[i-1] {
			return nil, nil, fmt.Errorf("invalid marker position %d", pos)
		}
	}

	h, err := w.encode(ids)
	if err != nil {
		return nil, nil, err
	}
	defer func() { C.mlx_array_free(h) }()
	length, width := len(ids), w.Config.HiddenSize

	// Add the question type, then run the decision Transformer blocks.
	g := &graph{w: w}
	kind := C.int32_t(qtype)
	typed := g.take(g.weight("type_emb.weight"), g.data(unsafe.Pointer(&kind), C.MLX_INT32, 1), 0)
	next := g.binary("add", h, g.shape(typed, 1, 1, width))
	kept, err := retain(g, next)
	g.close()
	if err != nil {
		return nil, nil, err
	}
	C.mlx_array_free(h)
	h = kept
	for layer := range w.Agent.HeadLayers {
		g = &graph{w: w}
		next = w.headLayer(g, h, layer, length)
		kept, saveErr := retain(g, next)
		g.close()
		if saveErr != nil {
			return nil, nil, fmt.Errorf("decision layer %d: %w", layer, saveErr)
		}
		C.mlx_array_free(h)
		h = kept
	}

	// Score the marker states and keep the first token for the action head.
	g = &graph{w: w}
	positions := make([]C.int32_t, len(markers))
	for i, pos := range markers {
		positions[i] = C.int32_t(pos)
	}
	at := g.take(h, g.data(unsafe.Pointer(&positions[0]), C.MLX_INT32, len(markers)), 1)
	score := g.normWithBias(at, "scorer.layers.0.weight", "scorer.layers.0.bias", 1e-5)
	score = g.linearBias(score, "scorer.layers.1")
	score = g.gelu(score)
	score = g.linearBias(score, "scorer.layers.3")
	logits, err := g.output(score)
	if err != nil {
		g.close()
		return nil, nil, err
	}
	first := g.slice(h, []int{0, 0, 0}, []int{1, 1, width})
	pooled, err := g.output(first)
	g.close()
	if err != nil {
		return nil, nil, err
	}

	// The action model consumes normalized option probabilities and CLS state.
	features := actionFeatures(logits)
	pooled = append(pooled, features[:]...)
	g = &graph{w: w}
	vector := g.cast(g.data(unsafe.Pointer(&pooled[0]), C.MLX_FLOAT32, 1, len(pooled)), C.MLX_FLOAT16)
	action := g.linearBias(vector, "act_head.layers.0")
	action = g.gelu(action)
	action = g.linearBias(action, "act_head.layers.2")
	out, err := g.output(action)
	g.close()
	runtime.KeepAlive(pooled)
	if err != nil {
		return nil, nil, err
	}
	return logits, out, nil
}

func retain(g *graph, a C.mlx_array) (C.mlx_array, error) {
	if g.err != nil {
		return C.mlx_array{}, g.err
	}
	if C.mlx_array_eval(a) != 0 {
		return C.mlx_array{}, fmt.Errorf("MLX evaluation failed")
	}
	var saved C.mlx_array
	if C.mlx_array_set(&saved, a) != 0 {
		return C.mlx_array{}, fmt.Errorf("cannot retain MLX array")
	}
	return saved, nil
}

func (w *Weights) headLayer(g *graph, h C.mlx_array, layer, length int) C.mlx_array {
	width := w.Config.HiddenSize
	heads := max(1, width/64)
	if width%heads != 0 {
		g.err = fmt.Errorf("invalid decision head count")
		return C.mlx_array{}
	}
	dim := width / heads
	base := fmt.Sprintf("head.layers.%d.", layer)

	// This attention is bidirectional and has no RoPE.
	x := g.normWithBias(h, base+"norm1.weight", base+"norm1.bias", 1e-5)
	qkv := g.linearBias(x, base+"self_attn.in_proj")
	qkv = g.shape(qkv, 1, length, 3, heads, dim)
	parts := [3]C.mlx_array{}
	for i := range parts {
		part := g.slice(qkv, []int{0, 0, i, 0, 0}, []int{1, length, i + 1, heads, dim})
		parts[i] = g.transpose(g.shape(part, 1, length, heads, dim), 0, 2, 1, 3)
	}
	var attn C.mlx_array
	if g.err == nil {
		mode := C.CString("")
		g.own(&attn, C.mlx_fast_scaled_dot_product_attention(&attn, parts[0], parts[1], parts[2], C.float(1/math.Sqrt(float64(dim))), mode, C.mlx_array{}, C.mlx_array{}, false, w.stream), "decision attention")
		C.free(unsafe.Pointer(mode))
	}
	attn = g.transpose(attn, 0, 2, 1, 3)
	attn = g.shape(attn, 1, length, width)
	h = g.binary("add", h, g.linearBias(attn, base+"self_attn.out_proj"))

	// PyTorch's TransformerEncoderLayer uses ReLU in this block.
	ff := g.normWithBias(h, base+"norm2.weight", base+"norm2.bias", 1e-5)
	ff = g.unary("relu", g.linearBias(ff, base+"linear1"))
	return g.binary("add", h, g.linearBias(ff, base+"linear2"))
}

func actionFeatures(logits []float32) [4]float32 {
	k := max(len(logits), 2)
	z := make([]float64, k)
	peak := float64(logits[0])
	for _, v := range logits {
		peak = max(peak, float64(v))
	}
	total := float64(0)
	for i, v := range logits {
		z[i] = math.Exp(float64(v) - peak)
		total += z[i]
	}
	for i := range z {
		z[i] /= total
	}
	best, second, entropy := float64(0), float64(0), float64(0)
	for _, p := range z {
		if p > best {
			second, best = best, p
		} else if p > second {
			second = p
		}
		if p > 0 {
			entropy -= p * math.Log(max(p, 1e-9))
		}
	}
	return [4]float32{float32(best), float32(best - second), float32(entropy / math.Log(float64(k))), float32(float64(k) / 255)}
}
