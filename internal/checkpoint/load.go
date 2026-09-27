package checkpoint

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
)

const maxHeader = 16 << 20

// Encoder holds the dimensions needed to match checkpoint weights.
type Encoder struct {
	ModelType         string   `json:"model_type"`
	VocabSize         int      `json:"vocab_size"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	MaxPositions      int      `json:"max_position_embeddings"`
	NormBias          bool     `json:"norm_bias"`
	AttentionBias     bool     `json:"attention_bias"`
	MLPBias           bool     `json:"mlp_bias"`
	Activation        string   `json:"hidden_activation"`
	NormEps           float32  `json:"norm_eps"`
	LocalAttention    int      `json:"local_attention"`
	GlobalEvery       int      `json:"global_attn_every_n_layers"`
	LayerTypes        []string `json:"layer_types"`
	Rope              map[string]struct {
		Theta float32 `json:"rope_theta"`
		Type  string  `json:"rope_type"`
	} `json:"rope_parameters"`
}

type Agent struct {
	Encoder    string             `json:"encoder"`
	HeadLayers int                `json:"head_layers"`
	MaxLen     int                `json:"max_len"`
	HeadMaxLen int                `json:"head_max_len"`
	ActCosts   map[string]float64 `json:"act_costs"`
}

type Format struct {
	Name    string `json:"format"`
	Version int    `json:"format_version"`
	Dtype   string `json:"dtype"`
}

type Tensor struct {
	Dtype string   `json:"dtype"`
	Shape []int    `json:"shape"`
	Range [2]int64 `json:"data_offsets"`
}

// Checkpoint keeps the weight file open for later tensor reads.
type Checkpoint struct {
	Encoder Encoder
	Agent   Agent
	Format  Format
	Tensors map[string]Tensor
	file    *os.File
	start   int64
}

func (c *Checkpoint) Close() error { return c.file.Close() }

// Read copies one tensor into dst. Callers own the returned bytes.
func (c *Checkpoint) Read(name string, dst []byte) error {
	t, ok := c.Tensors[name]
	if !ok {
		return fmt.Errorf("unknown tensor %q", name)
	}
	if int64(len(dst)) != t.Range[1]-t.Range[0] {
		return fmt.Errorf("tensor %q needs %d bytes", name, t.Range[1]-t.Range[0])
	}
	_, err := c.file.ReadAt(dst, c.start+t.Range[0])
	return err
}

// Load validates the configs and every weight before accepting a checkpoint.
func Load(dir string) (_ *Checkpoint, err error) {
	c := &Checkpoint{}
	for _, item := range []struct {
		name string
		out  any
	}{
		{"encoder/config.json", &c.Encoder},
		{"rl_agent_config.json", &c.Agent},
		{"mlx_config.json", &c.Format},
	} {
		data, readErr := os.ReadFile(filepath.Join(dir, item.name))
		if readErr != nil {
			return nil, fmt.Errorf("%s: %w", item.name, readErr)
		}
		if readErr = json.Unmarshal(data, item.out); readErr != nil {
			return nil, fmt.Errorf("%s: %w", item.name, readErr)
		}
	}
	if err := validateConfig(c); err != nil {
		return nil, err
	}

	c.file, err = os.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			c.file.Close()
		}
	}()

	if err = c.loadHeader(); err != nil {
		return nil, err
	}
	return c, nil
}

func validateConfig(c *Checkpoint) error {
	e, a, f := c.Encoder, c.Agent, c.Format
	if f.Name != "laya-mlx" || f.Version != 1 || f.Dtype != "float16" {
		return fmt.Errorf("unsupported checkpoint format: %+v", f)
	}
	if e.ModelType != "modernbert" || e.Activation != "gelu" || e.NormBias || e.AttentionBias || e.MLPBias ||
		e.VocabSize < 1 || e.VocabSize > 1<<20 || e.HiddenSize < 1 || e.HiddenSize > 8192 ||
		e.IntermediateSize < 1 || e.IntermediateSize > 32768 || e.NumHiddenLayers < 1 || e.NumHiddenLayers > 128 ||
		e.NumAttentionHeads < 1 || e.HiddenSize%e.NumAttentionHeads != 0 || (e.HiddenSize/e.NumAttentionHeads)%2 != 0 ||
		e.MaxPositions < 1 {
		return fmt.Errorf("unsupported encoder configuration")
	}
	if e.NormEps < 0 || math.IsNaN(float64(e.NormEps)) || e.LocalAttention < 0 || e.GlobalEvery < 0 ||
		(len(e.LayerTypes) != 0 && len(e.LayerTypes) != e.NumHiddenLayers) {
		return fmt.Errorf("invalid encoder attention settings")
	}
	for _, kind := range e.LayerTypes {
		if kind != "full_attention" && kind != "sliding_attention" {
			return fmt.Errorf("unsupported attention type %q", kind)
		}
	}
	for _, params := range e.Rope {
		if params.Type != "" && params.Type != "default" || params.Theta < 0 || math.IsNaN(float64(params.Theta)) {
			return fmt.Errorf("unsupported RoPE settings")
		}
	}
	if a.Encoder == "" || a.HeadLayers < 1 || a.HeadLayers > 32 ||
		len(a.ActCosts) != 1 || a.ActCosts["escalate"] == 0 ||
		a.MaxLen > e.MaxPositions || a.HeadMaxLen <= 4 || a.HeadMaxLen >= a.MaxLen {
		return fmt.Errorf("unsupported agent configuration")
	}
	return nil
}

func (c *Checkpoint) loadHeader() error {
	info, err := c.file.Stat()
	if err != nil {
		return err
	}
	var prefix [8]byte
	if _, err := io.ReadFull(c.file, prefix[:]); err != nil {
		return err
	}
	size := binary.LittleEndian.Uint64(prefix[:])
	if size == 0 || size > maxHeader || size > uint64(info.Size()-8) {
		return fmt.Errorf("invalid safetensors header length %d", size)
	}
	c.start = int64(size) + 8
	header := make([]byte, int(size))
	if _, err := io.ReadFull(c.file, header); err != nil {
		return err
	}
	if header[0] != '{' {
		return fmt.Errorf("invalid safetensors header")
	}
	if err := c.parseHeader(header); err != nil {
		return fmt.Errorf("safetensors header: %w", err)
	}
	return c.validateTensors(info.Size() - c.start)
}

func (c *Checkpoint) parseHeader(header []byte) error {
	dec := json.NewDecoder(bytes.NewReader(header))
	if _, err := dec.Token(); err != nil {
		return err
	}
	c.Tensors = make(map[string]Tensor)
	seen := make(map[string]bool)
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return err
		}
		name := key.(string)
		if seen[name] {
			return fmt.Errorf("duplicate tensor %q", name)
		}
		seen[name] = true
		if name == "__metadata__" {
			var metadata map[string]string
			if err := dec.Decode(&metadata); err != nil {
				return err
			}
			continue
		}
		var tensor Tensor
		if err := dec.Decode(&tensor); err != nil {
			return err
		}
		c.Tensors[name] = tensor
	}
	if _, err := dec.Token(); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return fmt.Errorf("trailing header data: %v", err)
	}
	return nil
}

func (c *Checkpoint) validateTensors(length int64) error {
	expected := shapes(c.Encoder, c.Agent)
	if len(c.Tensors) != len(expected) {
		return fmt.Errorf("tensor count: got %d, want %d", len(c.Tensors), len(expected))
	}
	type span struct {
		name     string
		from, to int64
	}
	spans := make([]span, 0, len(expected))
	for name, shape := range expected {
		t, ok := c.Tensors[name]
		if !ok {
			return fmt.Errorf("missing tensor %q", name)
		}
		bytes := int64(2)
		if len(shape) != len(t.Shape) {
			return fmt.Errorf("tensor %q has wrong shape", name)
		}
		for i, dim := range shape {
			if t.Shape[i] != dim {
				return fmt.Errorf("tensor %q has wrong shape", name)
			}
			bytes *= int64(dim)
		}
		if t.Dtype != "F16" || t.Range[0] < 0 || t.Range[1] < t.Range[0] ||
			t.Range[1] > length || t.Range[1]-t.Range[0] != bytes {
			return fmt.Errorf("tensor %q has invalid dtype or offsets", name)
		}
		spans = append(spans, span{name, t.Range[0], t.Range[1]})
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].from < spans[j].from })
	end := int64(0)
	for _, s := range spans {
		if s.from != end {
			return fmt.Errorf("gap or overlap at tensor %q", s.name)
		}
		end = s.to
	}
	if end != length {
		return fmt.Errorf("unindexed bytes after tensors")
	}
	return nil
}
