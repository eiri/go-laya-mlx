package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Embed runs ModernBERT's token lookup and embedding layer normalization.
// It returns row-major FP32 values with shape [len(ids), hidden_size].
func (w *Weights) Embed(ids []int) ([]float32, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || len(ids) == 0 {
		return nil, fmt.Errorf("closed model or empty token IDs")
	}
	input := make([]C.int32_t, len(ids))
	for i, id := range ids {
		if id < 0 || id >= w.Config.VocabSize {
			return nil, fmt.Errorf("token ID %d is outside vocabulary", id)
		}
		input[i] = C.int32_t(id)
	}

	// Each map lookup returns an owned handle to the stored weight.
	get := func(name string) (C.mlx_array, error) {
		key := C.CString(name)
		defer C.free(unsafe.Pointer(key))
		var arr C.mlx_array
		if C.mlx_map_string_to_array_get(&arr, w.values, key) != 0 {
			return arr, fmt.Errorf("missing weight %q", name)
		}
		return arr, nil
	}
	weight, err := get("encoder.embeddings.tok_embeddings.weight")
	if err != nil {
		return nil, err
	}
	defer C.mlx_array_free(weight)
	norm, err := get("encoder.embeddings.norm.weight")
	if err != nil {
		return nil, err
	}
	defer C.mlx_array_free(norm)

	shape := [1]C.int{C.int(len(ids))}
	indices := C.mlx_array_new_data(unsafe.Pointer(&input[0]), &shape[0], 1, C.MLX_INT32)
	defer C.mlx_array_free(indices)
	var selected C.mlx_array
	if C.mlx_take_axis(&selected, weight, indices, 0, w.stream) != 0 {
		return nil, fmt.Errorf("MLX embedding lookup failed")
	}
	defer C.mlx_array_free(selected)
	var normalized C.mlx_array
	if C.mlx_fast_layer_norm(&normalized, selected, norm, C.mlx_array{}, 1e-5, w.stream) != 0 {
		return nil, fmt.Errorf("MLX embedding norm failed")
	}
	defer C.mlx_array_free(normalized)
	var converted C.mlx_array
	if C.mlx_astype(&converted, normalized, C.MLX_FLOAT32, w.stream) != 0 {
		return nil, fmt.Errorf("MLX embedding cast failed")
	}
	defer C.mlx_array_free(converted)
	if C.mlx_array_eval(converted) != 0 {
		return nil, fmt.Errorf("MLX embedding eval failed")
	}

	count := int(C.mlx_array_size(converted))
	data := unsafe.Slice((*float32)(unsafe.Pointer(C.mlx_array_data_float32(converted))), count)
	result := append([]float32(nil), data...)
	runtime.KeepAlive(input)
	return result, nil
}
