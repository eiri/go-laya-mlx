package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"fmt"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/eiri/go-laya-mlx/internal/checkpoint"
)

// Weights owns MLX arrays loaded from a validated local checkpoint.
type Weights struct {
	mu       sync.Mutex
	values   C.mlx_map_string_to_array
	stream   C.mlx_stream
	ioStream C.mlx_stream
	Config   checkpoint.Encoder
	names    map[string]struct{}
	closed   bool
}

// LoadWeights reads the local Hub cache; it never downloads model files.
func LoadWeights() (_ *Weights, err error) {
	dir, err := checkpoint.Cached()
	if err != nil {
		return nil, err
	}
	c, err := checkpoint.Load(dir)
	if err != nil {
		return nil, err
	}
	defer c.Close()

	// A global stream allows locked calls from different Go threads.
	device := C.mlx_device_new_type(C.MLX_GPU, 0)
	defer C.mlx_device_free(device)
	w := &Weights{Config: c.Encoder, names: make(map[string]struct{}, len(c.Tensors))}
	for name := range c.Tensors {
		w.names[name] = struct{}{}
	}
	w.stream = C.mlx_stream_new_thread_unsafe(device)
	if w.stream.ctx == nil {
		return nil, fmt.Errorf("cannot create MLX GPU stream")
	}
	cpu := C.mlx_device_new_type(C.MLX_CPU, 0)
	defer C.mlx_device_free(cpu)
	w.ioStream = C.mlx_stream_new_thread_unsafe(cpu)
	if w.ioStream.ctx == nil {
		C.mlx_stream_free(w.stream)
		return nil, fmt.Errorf("cannot create MLX CPU stream")
	}
	w.values = C.mlx_map_string_to_array_new()
	metadata := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(metadata)
	path := C.CString(filepath.Join(dir, "model.safetensors"))
	defer C.free(unsafe.Pointer(path))
	// File reads must run on CPU; GPU evaluation of MLX load is unsupported.
	if C.mlx_load_safetensors(&w.values, &metadata, path, w.ioStream) != 0 {
		w.Close()
		return nil, fmt.Errorf("cannot load safetensors with MLX")
	}
	return w, nil
}

// Close releases the model arrays and its Metal stream.
func (w *Weights) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	C.mlx_map_string_to_array_free(w.values)
	C.mlx_stream_free(w.ioStream)
	C.mlx_stream_free(w.stream)
	return nil
}

// Tensor returns a copy of one evaluated tensor as FP32 values.
func (w *Weights) Tensor(name string) ([]float32, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil, fmt.Errorf("weights are closed")
	}
	if _, ok := w.names[name]; !ok {
		return nil, fmt.Errorf("unknown tensor %q", name)
	}
	key := C.CString(name)
	defer C.free(unsafe.Pointer(key))
	var value C.mlx_array
	if C.mlx_map_string_to_array_get(&value, w.values, key) != 0 {
		return nil, fmt.Errorf("unknown tensor %q", name)
	}
	defer C.mlx_array_free(value)
	var converted C.mlx_array
	if C.mlx_astype(&converted, value, C.MLX_FLOAT32, w.stream) != 0 {
		return nil, fmt.Errorf("cannot convert tensor %q", name)
	}
	defer C.mlx_array_free(converted)
	if C.mlx_array_eval(converted) != 0 {
		return nil, fmt.Errorf("cannot evaluate tensor %q", name)
	}
	count := int(C.mlx_array_size(converted))
	data := unsafe.Slice((*float32)(unsafe.Pointer(C.mlx_array_data_float32(converted))), count)
	result := append([]float32(nil), data...)
	runtime.KeepAlive(w)
	return result, nil
}
