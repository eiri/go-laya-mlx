package prompt

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func tinyTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	vocab := map[string]int{"Ġ": 95}
	for i := 32; i < 127; i++ {
		vocab[string(rune(i))] = i - 32
	}
	added := []addedToken{
		{ID: 128, Content: "[CLS]", Special: true},
		{ID: 129, Content: "[SEP]", Special: true},
		{ID: 130, Content: "[PAD]", Special: true},
		{ID: 131, Content: "[MASK]", Special: true, LStrip: true},
	}
	dir := t.TempDir()
	data, err := json.Marshal(map[string]any{
		"model":          map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
		"normalizer":     map[string]any{"type": "NFC"},
		"pre_tokenizer":  map[string]any{"type": "ByteLevel", "use_regex": true},
		"post_processor": map[string]any{"type": "TemplateProcessing", "single": []any{map[string]any{"SpecialToken": map[string]any{"id": "[CLS]"}}, map[string]any{"Sequence": map[string]any{"id": "A"}}, map[string]any{"SpecialToken": map[string]any{"id": "[SEP]"}}}},
		"added_tokens":   added,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"cls_token":"[CLS]","sep_token":"[SEP]","pad_token":"[PAD]","mask_token":"[MASK]"}`), 0600); err != nil {
		t.Fatal(err)
	}
	tok, err := NewTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestPrepare(t *testing.T) {
	tok := tinyTokenizer(t)
	q := Question{Type: "choice", Instructions: "Pick", Options: []Option{{Label: "yes"}, {Label: "no"}}}
	seq, err := Prepare(tok, "abc", q, 80, 48)
	if err != nil {
		t.Fatal(err)
	}
	if len(seq.Markers) != 2 || seq.IDs[0] != tok.CLS || seq.IDs[seq.Markers[0]] != tok.MASK || seq.IDs[seq.Markers[1]] != tok.MASK || seq.IDs[len(seq.IDs)-1] != tok.SEP {
		t.Fatalf("wrong sequence: %+v", seq)
	}
	if seq.StateDropped != 0 {
		t.Fatalf("dropped %d tokens", seq.StateDropped)
	}

	short, err := Prepare(tok, strings.Repeat("abc", 40), q, 48, 32)
	if err != nil {
		t.Fatal(err)
	}
	if len(short.IDs) != 48 || short.StateDropped == 0 {
		t.Fatalf("expected truncated state, got %+v", short)
	}
}

func TestNoulOptions(t *testing.T) {
	got, err := options(Question{Type: "noul"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"false: no, the statement does not hold", "true: yes, the statement holds"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestCachedTokens(t *testing.T) {
	dir := os.Getenv("LAYA_MODEL_DIR")
	if dir == "" {
		t.Skip("set LAYA_MODEL_DIR to test the cached checkpoint")
	}
	tok, err := NewTokenizer(filepath.Join(dir, "tokenizer"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		text string
		want []int
	}{
		{"Hello world!", []int{12092, 1533, 2}},
		{"choice question: Which team should handle this request?", []int{22122, 1953, 27, 6758, 2285, 943, 6016, 436, 2748, 32}},
		{" billing: invoices, payments, refunds", []int{33484, 27, 29838, 1271, 13, 10762, 13, 1275, 41748}},
		{"   hello  world\n", []int{50275, 25521, 50276, 10186, 187}},
		{"你好世界", []int{24553, 34439, 42848, 45261}},
	} {
		got, err := tok.Encode(tc.text)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%q: got %v, want %v", tc.text, got, tc.want)
		}
	}
	for _, tc := range []struct {
		q       Question
		ids     []int
		markers []int
	}{
		{Question{Type: "choice", Instructions: "Which team?", Options: []Option{{"billing", "refunds"}, {"tech", "bugs"}}}, []int{50281, 22122, 1953, 27, 6758, 2285, 32, 50282, 50284, 33484, 27, 1275, 41748, 50284, 13817, 27, 19775, 50282, 42, 369, 47045, 7019, 15, 50282}, []int{8, 13}},
		{Question{Type: "score", Instructions: "How urgent?", Levels: []any{"later", "now"}}, []int{50281, 18891, 1953, 27, 1359, 21007, 32, 50282, 50284, 1268, 470, 27, 1996, 50284, 1268, 337, 27, 1024, 50282, 42, 369, 47045, 7019, 15, 50282}, []int{8, 13}},
		{Question{Type: "noul", Instructions: "Does the customer ask for a refund?"}, []int{50281, 79, 3941, 1953, 27, 9876, 253, 7731, 1642, 323, 247, 23005, 32, 50282, 50284, 3221, 27, 642, 13, 253, 3908, 1057, 417, 2186, 50284, 2032, 27, 4754, 13, 253, 3908, 6556, 50282, 42, 369, 47045, 7019, 15, 50282}, []int{14, 24}},
	} {
		seq, err := Prepare(tok, "I was billed twice.", tc.q, 512, 192)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(seq.IDs, tc.ids) || !reflect.DeepEqual(seq.Markers, tc.markers) {
			t.Errorf("%s: got IDs %v markers %v; want IDs %v markers %v", tc.q.Type, seq.IDs, seq.Markers, tc.ids, tc.markers)
		}
	}
}
