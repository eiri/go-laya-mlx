package checkpoint

import "fmt"

// shapes follows the parameter names and matrix layouts in Laya-MLX.
func shapes(e Encoder, a Agent) map[string][]int {
	h, m := e.HiddenSize, e.IntermediateSize
	result := map[string][]int{
		"encoder.embeddings.tok_embeddings.weight": {e.VocabSize, h},
		"encoder.embeddings.norm.weight":           {h},
		"encoder.final_norm.weight":                {h},
		"type_emb.weight":                          {3, h},
		"temperature":                              {3},
		"scorer.layers.0.weight":                   {h},
		"scorer.layers.0.bias":                     {h},
		"scorer.layers.1.weight":                   {h, h},
		"scorer.layers.1.bias":                     {h},
		"scorer.layers.3.weight":                   {1, h},
		"scorer.layers.3.bias":                     {1},
		"act_head.layers.0.weight":                 {256, h + 4},
		"act_head.layers.0.bias":                   {256},
		"act_head.layers.2.weight":                 {2, 256},
		"act_head.layers.2.bias":                   {2},
	}

	for i := range e.NumHiddenLayers {
		base := fmt.Sprintf("encoder.layers.%d.", i)
		result[base+"attn.Wqkv.weight"] = []int{3 * h, h}
		result[base+"attn.Wo.weight"] = []int{h, h}
		result[base+"mlp.Wi.weight"] = []int{2 * m, h}
		result[base+"mlp.Wo.weight"] = []int{h, m}
		result[base+"mlp_norm.weight"] = []int{h}
		if i > 0 {
			result[base+"attn_norm.weight"] = []int{h}
		}
	}

	for i := range a.HeadLayers {
		base := fmt.Sprintf("head.layers.%d.", i)
		for _, norm := range []string{"norm1", "norm2"} {
			result[base+norm+".weight"] = []int{h}
			result[base+norm+".bias"] = []int{h}
		}
		result[base+"self_attn.in_proj.weight"] = []int{3 * h, h}
		result[base+"self_attn.in_proj.bias"] = []int{3 * h}
		result[base+"self_attn.out_proj.weight"] = []int{h, h}
		result[base+"self_attn.out_proj.bias"] = []int{h}
		result[base+"linear1.weight"] = []int{4 * h, h}
		result[base+"linear1.bias"] = []int{4 * h}
		result[base+"linear2.weight"] = []int{h, 4 * h}
		result[base+"linear2.bias"] = []int{h}
	}
	return result
}
