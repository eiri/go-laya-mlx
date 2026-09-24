package prompt

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	bpe "github.com/surya-mp/go-tokenizer"
	"golang.org/x/text/unicode/norm"
)

type addedToken struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	Normalized bool   `json:"normalized"`
	Special    bool   `json:"special"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	SingleWord bool   `json:"single_word"`
}

// Tokenizer handles the ModernBERT tokenizer.json used by this checkpoint.
type Tokenizer struct {
	model    *bpe.ByteLevelBPE
	added    []addedToken
	CLS      int
	SEP      int
	PAD      int
	MASK     int
	MaskText string
}

func NewTokenizer(dir string) (*Tokenizer, error) {
	path := filepath.Join(dir, "tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Model struct {
			Type string `json:"type"`
		} `json:"model"`
		Normalizer struct {
			Type string `json:"type"`
		} `json:"normalizer"`
		Pre struct {
			Type   string `json:"type"`
			Prefix bool   `json:"add_prefix_space"`
			Regex  bool   `json:"use_regex"`
		} `json:"pre_tokenizer"`
		Post struct {
			Type   string            `json:"type"`
			Single []json.RawMessage `json:"single"`
		} `json:"post_processor"`
		Added []addedToken `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if doc.Model.Type != "BPE" || doc.Normalizer.Type != "NFC" || doc.Pre.Type != "ByteLevel" || doc.Pre.Prefix || !doc.Pre.Regex || doc.Post.Type != "TemplateProcessing" || len(doc.Post.Single) != 3 {
		return nil, fmt.Errorf("unsupported tokenizer pipeline")
	}
	for _, token := range doc.Added {
		if token.Content == "" || token.SingleWord || token.RStrip || (token.LStrip && token.Content != "[MASK]") {
			return nil, fmt.Errorf("unsupported added token %q", token.Content)
		}
	}
	model, err := bpe.LoadBPE(path)
	if err != nil {
		return nil, err
	}
	wrapped, err := bpe.NewByteLevelBPE(model)
	if err != nil {
		return nil, err
	}
	config, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}
	var names struct {
		CLS  string `json:"cls_token"`
		SEP  string `json:"sep_token"`
		PAD  string `json:"pad_token"`
		MASK string `json:"mask_token"`
	}
	if err := json.Unmarshal(config, &names); err != nil {
		return nil, err
	}
	ids := make(map[string]int, len(doc.Added))
	for _, token := range doc.Added {
		if _, ok := ids[token.Content]; ok {
			return nil, fmt.Errorf("duplicate added token %q", token.Content)
		}
		ids[token.Content] = token.ID
	}
	for _, name := range []string{names.CLS, names.SEP, names.PAD, names.MASK} {
		if _, ok := ids[name]; !ok || name == "" {
			return nil, fmt.Errorf("missing tokenizer control token %q", name)
		}
	}
	var start, middle, end map[string]struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(doc.Post.Single[0], &start) != nil || start["SpecialToken"].ID != names.CLS ||
		json.Unmarshal(doc.Post.Single[1], &middle) != nil || middle["Sequence"].ID != "A" ||
		json.Unmarshal(doc.Post.Single[2], &end) != nil || end["SpecialToken"].ID != names.SEP {
		return nil, fmt.Errorf("unsupported tokenizer template")
	}
	// Match longer added tokens first when several start at the same byte.
	sort.SliceStable(doc.Added, func(i, j int) bool { return len(doc.Added[i].Content) > len(doc.Added[j].Content) })
	return &Tokenizer{model: wrapped, added: doc.Added, CLS: ids[names.CLS], SEP: ids[names.SEP], PAD: ids[names.PAD], MASK: ids[names.MASK], MaskText: names.MASK}, nil
}

// Encode omits the template's CLS and SEP tokens; sequence construction adds them.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	text = norm.NFC.String(text)
	var ids []int
	var plain strings.Builder
	flush := func() error {
		chunk, err := (bpe.ByteLevel{}).Encode(t.model, plain.String())
		ids = append(ids, chunk...)
		plain.Reset()
		return err
	}
	for len(text) > 0 {
		var match *addedToken
		for i := range t.added {
			if strings.HasPrefix(text, t.added[i].Content) {
				match = &t.added[i]
				break
			}
		}
		if match != nil {
			if match.LStrip {
				chunk := strings.TrimRightFunc(plain.String(), unicode.IsSpace)
				plain.Reset()
				plain.WriteString(chunk)
			}
			if err := flush(); err != nil {
				return nil, err
			}
			ids = append(ids, match.ID)
			text = text[len(match.Content):]
			continue
		}
		_, n := utf8.DecodeRuneInString(text)
		plain.WriteString(text[:n])
		text = text[n:]
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return ids, nil
}
