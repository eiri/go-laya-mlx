package mlx

/*
#include "mlx/c/mlx.h"
*/
import "C"

import "fmt"

// Memory reports active and peak MLX allocations in bytes.
func Memory() (active, peak uint64, err error) {
	var used, high C.size_t
	if C.mlx_get_active_memory(&used) != 0 || C.mlx_get_peak_memory(&high) != 0 {
		return 0, 0, fmt.Errorf("cannot read MLX memory")
	}
	return uint64(used), uint64(high), nil
}

// ResetPeak clears unused MLX buffers before a new measurement.
func ResetPeak() error {
	if C.mlx_clear_cache() != 0 || C.mlx_reset_peak_memory() != 0 {
		return fmt.Errorf("cannot reset MLX memory")
	}
	return nil
}
