package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// graph owns intermediate handles for one forward pass.
type graph struct {
	w      *Weights
	arrays []C.mlx_array
	err    error
}

func (g *graph) close() {
	for _, a := range g.arrays {
		C.mlx_array_free(a)
	}
}

func (g *graph) own(a *C.mlx_array, status C.int, op string) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	if status != 0 {
		g.err = fmt.Errorf("MLX %s failed", op)
		return C.mlx_array{}
	}
	g.arrays = append(g.arrays, *a)
	return *a
}

func (g *graph) weight(name string) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	key := C.CString(name)
	defer C.free(unsafe.Pointer(key))
	var a C.mlx_array
	return g.own(&a, C.mlx_map_string_to_array_get(&a, g.w.values, key), name)
}

func (g *graph) data(data unsafe.Pointer, dtype C.mlx_dtype, shape ...int) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	cdims := make([]C.int, len(shape))
	for i, dim := range shape {
		cdims[i] = C.int(dim)
	}
	var ptr *C.int
	if len(cdims) > 0 {
		ptr = &cdims[0]
	}
	a := C.mlx_array_new_data(data, ptr, C.int(len(cdims)), dtype)
	if a.ctx == nil {
		g.err = fmt.Errorf("MLX array allocation failed")
		return a
	}
	g.arrays = append(g.arrays, a)
	return a
}

func (g *graph) binary(op string, a, b C.mlx_array) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	var out C.mlx_array
	var status C.int
	switch op {
	case "add":
		status = C.mlx_add(&out, a, b, g.w.stream)
	case "mul":
		status = C.mlx_multiply(&out, a, b, g.w.stream)
	case "div":
		status = C.mlx_divide(&out, a, b, g.w.stream)
	default:
		g.err = fmt.Errorf("unknown operation %s", op)
		return out
	}
	return g.own(&out, status, op)
}

func (g *graph) unary(op string, a C.mlx_array) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	var out C.mlx_array
	var status C.int
	switch op {
	case "erf":
		status = C.mlx_erf(&out, a, g.w.stream)
	case "relu":
		status = C.mlx_maximum(&out, a, g.scalar(0), g.w.stream)
	default:
		g.err = fmt.Errorf("unknown operation %s", op)
		return out
	}
	return g.own(&out, status, op)
}

func (g *graph) scalar(value float32) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	a := C.mlx_array_new_float32(C.float(value))
	g.arrays = append(g.arrays, a)
	return a
}

func (g *graph) shape(a C.mlx_array, dims ...int) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	cdims := make([]C.int, len(dims))
	for i, dim := range dims {
		cdims[i] = C.int(dim)
	}
	var out C.mlx_array
	return g.own(&out, C.mlx_reshape(&out, a, &cdims[0], C.size_t(len(cdims)), g.w.stream), "reshape")
}

func (g *graph) transpose(a C.mlx_array, axes ...int) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	cdims := make([]C.int, len(axes))
	for i, dim := range axes {
		cdims[i] = C.int(dim)
	}
	var out C.mlx_array
	return g.own(&out, C.mlx_transpose_axes(&out, a, &cdims[0], C.size_t(len(cdims)), g.w.stream), "transpose")
}

func (g *graph) slice(a C.mlx_array, starts, ends []int) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	start, end, stride := make([]C.int, len(starts)), make([]C.int, len(starts)), make([]C.int, len(starts))
	for i := range starts {
		start[i], end[i], stride[i] = C.int(starts[i]), C.int(ends[i]), 1
	}
	var out C.mlx_array
	return g.own(&out, C.mlx_slice(&out, a, &start[0], C.size_t(len(start)), &end[0], C.size_t(len(end)), &stride[0], C.size_t(len(stride)), g.w.stream), "slice")
}

func (g *graph) take(a, indices C.mlx_array, axis int) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	var out C.mlx_array
	return g.own(&out, C.mlx_take_axis(&out, a, indices, C.int(axis), g.w.stream), "take")
}

func (g *graph) norm(a C.mlx_array, weight string, eps float32) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	w := g.weight(weight)
	var out C.mlx_array
	return g.own(&out, C.mlx_fast_layer_norm(&out, a, w, C.mlx_array{}, C.float(eps), g.w.stream), "layer norm")
}

func (g *graph) linear(a C.mlx_array, weight string) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	w := g.weight(weight)
	var transposed, out C.mlx_array
	g.own(&transposed, C.mlx_transpose(&transposed, w, g.w.stream), "transpose weight")
	if g.err != nil {
		return C.mlx_array{}
	}
	return g.own(&out, C.mlx_matmul(&out, a, transposed, g.w.stream), "matmul")
}

func (g *graph) gelu(a C.mlx_array) C.mlx_array {
	if g.err != nil {
		return C.mlx_array{}
	}
	// Match mlx.nn.gelu: x * (1 + erf(x / sqrt(2))) / 2.
	v := g.binary("div", a, g.scalar(1.4142135623730951))
	v = g.unary("erf", v)
	v = g.binary("add", v, g.scalar(1))
	v = g.binary("mul", a, v)
	return g.binary("div", v, g.scalar(2))
}

func (g *graph) output(a C.mlx_array) ([]float32, error) {
	if g.err != nil {
		return nil, g.err
	}
	var fp32 C.mlx_array
	g.own(&fp32, C.mlx_astype(&fp32, a, C.MLX_FLOAT32, g.w.stream), "cast")
	if g.err != nil {
		return nil, g.err
	}
	if C.mlx_array_eval(fp32) != 0 {
		return nil, fmt.Errorf("MLX evaluation failed")
	}
	count := int(C.mlx_array_size(fp32))
	data := unsafe.Slice((*float32)(unsafe.Pointer(C.mlx_array_data_float32(fp32))), count)
	return append([]float32(nil), data...), nil
}
