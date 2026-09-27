package checkpoint

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func fixture(t *testing.T, change func(map[string]Tensor)) string {
	t.Helper()
	dir := t.TempDir()
	e := Encoder{ModelType: "modernbert", VocabSize: 8, HiddenSize: 4, IntermediateSize: 8,
		NumHiddenLayers: 1, NumAttentionHeads: 1, MaxPositions: 512, Activation: "gelu"}
	a := Agent{Encoder: "ModernBERT", HeadLayers: 1, MaxLen: 128, HeadMaxLen: 32, ActCosts: map[string]float64{"escalate": 0.5}}
	f := Format{Name: "laya-mlx", Version: 1, Dtype: "float16"}
	for name, value := range map[string]any{
		"encoder/config.json": e, "rl_agent_config.json": a, "mlx_config.json": f,
	} {
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0644); err != nil {
			t.Fatal(err)
		}
	}

	tensors := make(map[string]Tensor)
	var offset int64
	for name, shape := range shapes(e, a) {
		size := int64(2)
		for _, dim := range shape {
			size *= int64(dim)
		}
		tensors[name] = Tensor{Dtype: "F16", Shape: shape, Range: [2]int64{offset, offset + size}}
		offset += size
	}
	if change != nil {
		change(tensors)
	}
	header, err := json.Marshal(tensors)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(header))); err != nil {
		t.Fatal(err)
	}
	buf.Write(header)
	buf.Write(make([]byte, offset))
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestLoad(t *testing.T) {
	c, err := Load(fixture(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if len(c.Tensors) != 32 {
		t.Fatalf("got %d tensors", len(c.Tensors))
	}
	tensor := c.Tensors["temperature"]
	if err := c.Read("temperature", make([]byte, tensor.Range[1]-tensor.Range[0])); err != nil {
		t.Fatal(err)
	}
	if err := c.Read("temperature", nil); err == nil {
		t.Fatal("expected buffer length error")
	}
}

func TestInvalidTensor(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(map[string]Tensor)
	}{
		{"missing", func(m map[string]Tensor) { delete(m, "temperature") }},
		{"shape", func(m map[string]Tensor) { v := m["temperature"]; v.Shape = []int{4}; m["temperature"] = v }},
		{"dtype", func(m map[string]Tensor) { v := m["temperature"]; v.Dtype = "F32"; m["temperature"] = v }},
		{"range", func(m map[string]Tensor) { v := m["temperature"]; v.Range[1]++; m["temperature"] = v }},
		{"overlap", func(m map[string]Tensor) { v := m["temperature"]; v.Range[0]++; v.Range[1]++; m["temperature"] = v }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Load(fixture(t, tc.edit)); err == nil {
				t.Fatal("accepted invalid checkpoint")
			}
		})
	}
}

func TestInvalidTemperature(t *testing.T) {
	dir := fixture(t, nil)
	path := filepath.Join(dir, "rl_agent_config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var agent Agent
	if err := json.Unmarshal(data, &agent); err != nil {
		t.Fatal(err)
	}
	agent.TemperatureByOpt = map[string]float64{"choice:11+": -1}
	data, err = json.Marshal(agent)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir); err == nil {
		t.Fatal("accepted invalid temperature")
	}
}

func TestDuplicateHeader(t *testing.T) {
	var c Checkpoint
	if err := c.parseHeader([]byte(`{"temperature":{},"temperature":{}}`)); err == nil {
		t.Fatal("accepted duplicate tensor name")
	}
}

func TestTruncated(t *testing.T) {
	dir := fixture(t, nil)
	path := filepath.Join(dir, "model.safetensors")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, info.Size()-1); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir); err == nil {
		t.Fatal("accepted truncated checkpoint")
	}
}

func TestLocalCheckpoint(t *testing.T) {
	dir, err := Cached()
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("aac6fef/laya-mlx is not cached")
	}
	if err != nil {
		t.Fatal(err)
	}
	c, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if got := len(c.Tensors); got != 206 {
		t.Fatalf("got %d tensors, want 206", got)
	}
	buf := make([]byte, 6)
	if err := c.Read("temperature", buf); err != nil {
		t.Fatal(err)
	}
	if c.Format.Name != "laya-mlx" {
		t.Fatal("wrong format")
	}
}
