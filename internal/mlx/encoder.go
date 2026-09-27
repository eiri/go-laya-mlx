package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// Encode runs ModernBERT on one unpadded sequence and returns row-major FP32.
func (w *Weights) Encode(ids []int) ([]float32, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	x, err := w.encode(ids)
	if err != nil {
		return nil, err
	}
	defer C.mlx_array_free(x)
	g := &graph{w: w}
	defer g.close()
	return g.output(x)
}

// encode keeps the final hidden states in FP16 for the decision layers.
// The caller holds w.mu and owns the returned array.
func (w *Weights) encode(ids []int) (C.mlx_array, error) {
	if w.closed || len(ids) == 0 || len(ids) > w.Config.MaxPositions {
		return C.mlx_array{}, fmt.Errorf("model closed or invalid sequence length")
	}
	input := make([]C.int32_t, len(ids))
	for i, id := range ids {
		if id < 0 || id >= w.Config.VocabSize {
			return C.mlx_array{}, fmt.Errorf("invalid token ID %d", id)
		}
		input[i] = C.int32_t(id)
	}
	g := &graph{w: w}
	idx := g.data(unsafe.Pointer(&input[0]), C.MLX_INT32, 1, len(ids))
	x := g.take(g.weight("encoder.embeddings.tok_embeddings.weight"), idx, 0)
	x = g.norm(x, "encoder.embeddings.norm.weight", w.normEps())
	if g.err != nil {
		g.close()
		return C.mlx_array{}, g.err
	}
	var kept C.mlx_array
	if C.mlx_array_set(&kept, x) != 0 {
		g.close()
		return kept, fmt.Errorf("cannot retain embeddings")
	}
	g.close()
	x = kept

	// Evaluate each block before freeing its graph to limit memory use.
	for layer := range w.Config.NumHiddenLayers {
		g = &graph{w: w}
		next := w.encLayer(g, x, layer, len(ids))
		if g.err == nil && C.mlx_array_eval(next) != 0 {
			g.err = fmt.Errorf("encoder layer %d evaluation failed", layer)
		}
		if g.err != nil {
			g.close()
			C.mlx_array_free(x)
			return C.mlx_array{}, g.err
		}
		var saved C.mlx_array
		if C.mlx_array_set(&saved, next) != 0 {
			g.close()
			C.mlx_array_free(x)
			return saved, fmt.Errorf("cannot retain encoder layer %d", layer)
		}
		g.close()
		C.mlx_array_free(x)
		x = saved
	}

	g = &graph{w: w}
	normalized := g.norm(x, "encoder.final_norm.weight", w.normEps())
	if g.err == nil && C.mlx_array_eval(normalized) != 0 {
		g.err = fmt.Errorf("encoder final normalization failed")
	}
	if g.err != nil {
		g.close()
		C.mlx_array_free(x)
		return C.mlx_array{}, g.err
	}
	var result C.mlx_array
	if C.mlx_array_set(&result, normalized) != 0 {
		g.close()
		C.mlx_array_free(x)
		return result, fmt.Errorf("cannot retain encoder output")
	}
	g.close()
	C.mlx_array_free(x)
	return result, nil
}

func (w *Weights) normEps() float32 {
	if w.Config.NormEps > 0 {
		return w.Config.NormEps
	}
	return 1e-5
}

func (w *Weights) encLayer(g *graph, x C.mlx_array, layer, length int) C.mlx_array {
	h, heads := w.Config.HiddenSize, w.Config.NumAttentionHeads
	dim := h / heads
	prefix := fmt.Sprintf("encoder.layers.%d.", layer)
	kind := "sliding_attention"
	if len(w.Config.LayerTypes) > 0 {
		kind = w.Config.LayerTypes[layer]
	} else if layer%max(1, w.Config.GlobalEvery) == 0 {
		kind = "full_attention"
	}

	// The first attention block has no input normalization.
	attnInput := x
	if layer > 0 {
		attnInput = g.norm(x, prefix+"attn_norm.weight", w.normEps())
	}
	qkv := g.linear(attnInput, prefix+"attn.Wqkv.weight")
	qkv = g.shape(qkv, 1, length, 3, heads, dim)
	parts := [3]C.mlx_array{}
	for k := range parts {
		part := g.slice(qkv, []int{0, 0, k, 0, 0}, []int{1, length, k + 1, heads, dim})
		parts[k] = g.transpose(g.shape(part, 1, length, heads, dim), 0, 2, 1, 3)
	}
	base := float32(10000)
	if kind == "full_attention" {
		base = 160000
	}
	if params, ok := w.Config.Rope[kind]; ok && params.Theta != 0 {
		base = params.Theta
	}
	for k := 0; k < 2; k++ {
		var rotated C.mlx_array
		g.own(&rotated, C.mlx_fast_rope(&rotated, parts[k], C.int(dim), false, C.mlx_optional_float{value: C.float(base), has_value: true}, 1, 0, C.mlx_array{}, w.stream), "RoPE")
		parts[k] = rotated
	}
	maskMode := C.CString("")
	mask := C.mlx_array{}
	if kind == "sliding_attention" {
		C.free(unsafe.Pointer(maskMode))
		maskMode = C.CString("array")
		window := w.Config.LocalAttention
		if window == 0 {
			window = 128
		}
		bits := make([]byte, length*length)
		for i := range length {
			for j := range length {
				if abs(i-j) <= window/2 {
					bits[i*length+j] = 1
				}
			}
		}
		mask = g.data(unsafe.Pointer(&bits[0]), C.MLX_BOOL, 1, 1, length, length)
	}
	var attention C.mlx_array
	if g.err == nil {
		g.own(&attention, C.mlx_fast_scaled_dot_product_attention(&attention, parts[0], parts[1], parts[2], C.float(1/math.Sqrt(float64(dim))), maskMode, mask, C.mlx_array{}, false, w.stream), "attention")
	}
	C.free(unsafe.Pointer(maskMode))
	if g.err != nil {
		return C.mlx_array{}
	}
	attention = g.transpose(attention, 0, 2, 1, 3)
	attention = g.shape(attention, 1, length, h)
	attention = g.linear(attention, prefix+"attn.Wo.weight")
	x = g.binary("add", x, attention)

	// ModernBERT uses a gated GELU feed-forward block.
	input := g.norm(x, prefix+"mlp_norm.weight", w.normEps())
	pair := g.linear(input, prefix+"mlp.Wi.weight")
	m := w.Config.IntermediateSize
	value := g.slice(pair, []int{0, 0, 0}, []int{1, length, m})
	gate := g.slice(pair, []int{0, 0, m}, []int{1, length, 2 * m})
	activation := g.binary("mul", g.gelu(value), gate)
	return g.binary("add", x, g.linear(activation, prefix+"mlp.Wo.weight"))
}

func abs(value int) int {
	if value < 0 {
		return -value
	}
	return value
}
