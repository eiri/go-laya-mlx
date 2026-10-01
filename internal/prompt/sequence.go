package prompt

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// Option keeps the caller's choice order, which determines marker order.
type Option struct {
	Label       string
	Description any
}

type Question struct {
	Type         string
	Instructions string
	Options      []Option
	Levels       []any
	False        any
	True         any
	FalseLabel   string
	TrueLabel    string
}

type Sequence struct {
	IDs          []int
	Markers      []int
	StateTokens  int
	StateDropped int
}

func serialize(value any) (string, error) {
	if text, ok := value.(string); ok {
		return text, nil
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(value); err != nil {
		return "", err
	}
	// Python's json.dumps adds a space after each separator.
	data := strings.TrimSuffix(out.String(), "\n")
	var spaced strings.Builder
	quoted, escaped := false, false
	for _, ch := range data {
		spaced.WriteRune(ch)
		if quoted {
			if escaped {
				escaped = false
			} else if ch == '\\' {
				escaped = true
			} else if ch == '"' {
				quoted = false
			}
		} else if ch == '"' {
			quoted = true
		} else if ch == ',' || ch == ':' {
			spaced.WriteByte(' ')
		}
	}
	return spaced.String(), nil
}

func empty(value any) bool {
	if value == nil {
		return true
	}
	text, ok := value.(string)
	return ok && text == ""
}

func criterion(value any) (string, error) {
	if text, ok := value.(string); ok {
		return text, nil
	}
	return serialize(value)
}

func options(q Question) ([]string, error) {
	switch q.Type {
	case "choice":
		if len(q.Options) == 0 {
			return nil, fmt.Errorf("choice needs options")
		}
		out := make([]string, 0, len(q.Options))
		for _, opt := range q.Options {
			if opt.Label == "" {
				return nil, fmt.Errorf("empty option label")
			}
			if empty(opt.Description) {
				out = append(out, opt.Label)
				continue
			}
			text, err := criterion(opt.Description)
			if err != nil {
				return nil, err
			}
			out = append(out, opt.Label+": "+text)
		}
		return out, nil
	case "score":
		if len(q.Levels) == 0 {
			return nil, fmt.Errorf("score needs levels")
		}
		out := make([]string, len(q.Levels))
		for i, level := range q.Levels {
			text, err := criterion(level)
			if err != nil {
				return nil, err
			}
			out[i] = fmt.Sprintf("level %d: %s", i, text)
		}
		return out, nil
	case "noul":
		falseLabel, trueLabel := q.FalseLabel, q.TrueLabel
		if falseLabel == "" {
			falseLabel = "false"
		}
		if trueLabel == "" {
			trueLabel = "true"
		}
		if falseLabel == trueLabel {
			return nil, fmt.Errorf("noul labels must differ")
		}
		if empty(q.False) {
			q.False = "no, the statement does not hold"
		}
		if empty(q.True) {
			q.True = "yes, the statement holds"
		}
		no, err := criterion(q.False)
		if err != nil {
			return nil, err
		}
		yes, err := criterion(q.True)
		if err != nil {
			return nil, err
		}
		return []string{falseLabel + ": " + no, trueLabel + ": " + yes}, nil
	default:
		return nil, fmt.Errorf("unknown question type %q", q.Type)
	}
}

// Prepare matches Laya's question prefix, marker positions, and state budget.
func Prepare(tok *Tokenizer, state any, q Question, maxLen, headMaxLen int) (Sequence, error) {
	if tok == nil || maxLen < 1 || headMaxLen < 1 || q.Instructions == "" {
		return Sequence{}, fmt.Errorf("invalid prompt settings")
	}
	texts, err := options(q)
	if err != nil {
		return Sequence{}, err
	}
	ins := strings.ReplaceAll(q.Instructions, tok.MaskText, " ")
	head, err := tok.Encode(q.Type + " question: " + ins)
	if err != nil {
		return Sequence{}, err
	}
	opts := make([][]int, len(texts))
	budget := headMaxLen
	for i, text := range texts {
		ids, err := tok.Encode(" " + strings.ReplaceAll(text, tok.MaskText, " "))
		if err != nil {
			return Sequence{}, err
		}
		if len(ids) > 48 {
			ids = ids[:48]
		}
		opts[i] = append([]int{tok.MASK}, ids...)
		budget -= len(opts[i])
	}
	if budget < 16 {
		per := max(4, (headMaxLen-16)/len(opts))
		budget = headMaxLen
		for i, ids := range opts {
			if len(ids) > per {
				opts[i] = ids[:per]
			}
			budget -= len(opts[i])
		}
	}
	if len(head) > max(8, budget) {
		head = head[:max(8, budget)]
	}
	ids := append([]int{tok.CLS}, head...)
	ids = append(ids, tok.SEP)
	markers := make([]int, len(opts))
	for i, opt := range opts {
		markers[i] = len(ids)
		ids = append(ids, opt...)
	}
	ids = append(ids, tok.SEP)

	text, err := serialize(state)
	if err != nil {
		return Sequence{}, err
	}
	stateIDs, err := tok.Encode(strings.ReplaceAll(text, tok.MaskText, " "))
	if err != nil {
		return Sequence{}, err
	}
	room := max(0, maxLen-len(ids)-1)
	kept := stateIDs
	if len(kept) > room {
		kind := reflect.ValueOf(state).Kind()
		if kind == reflect.Slice || kind == reflect.Array {
			kept = kept[len(kept)-room:]
		} else {
			kept = kept[:room]
		}
	}
	ids = append(ids, kept...)
	ids = append(ids, tok.SEP)
	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}
	var valid []int
	for _, marker := range markers {
		if marker < maxLen {
			valid = append(valid, marker)
		}
	}
	if len(valid) != len(markers) {
		return Sequence{}, fmt.Errorf("too many options for token budget")
	}
	return Sequence{IDs: ids, Markers: valid, StateTokens: len(stateIDs), StateDropped: len(stateIDs) - len(kept)}, nil
}
