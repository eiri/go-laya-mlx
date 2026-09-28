package mlx

/*
#cgo darwin,arm64 LDFLAGS: -lmlxc
#include "mlx/c/mlx.h"
*/
import "C"

import "fmt"

// add checks that Go can run MLX operations on the Metal device.
func add(a, b float32) (float32, error) {
	stream := C.mlx_default_gpu_stream_new()
	defer C.mlx_stream_free(stream)

	left := C.mlx_array_new_float32(C.float(a))
	defer C.mlx_array_free(left)
	right := C.mlx_array_new_float32(C.float(b))
	defer C.mlx_array_free(right)

	var result C.mlx_array
	if C.mlx_add(&result, left, right, stream) != 0 {
		return 0, fmt.Errorf("mlx_add failed")
	}
	defer C.mlx_array_free(result)

	if C.mlx_array_eval(result) != 0 {
		return 0, fmt.Errorf("mlx_array_eval failed")
	}

	var value C.float
	if C.mlx_array_item_float32(&value, result) != 0 {
		return 0, fmt.Errorf("mlx_array_item_float32 failed")
	}
	return float32(value), nil
}
