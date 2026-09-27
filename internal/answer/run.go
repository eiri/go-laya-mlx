package answer

import (
	"fmt"

	"github.com/eiri/go-laya-mlx/internal/mlx"
	"github.com/eiri/go-laya-mlx/internal/prompt"
)

// Run prepares one question, runs the model, and calibrates its answer.
func Run(model *mlx.Weights, tok *prompt.Tokenizer, state any, q prompt.Question) (Answer, prompt.Sequence, error) {
	if model == nil || tok == nil {
		return Answer{}, prompt.Sequence{}, fmt.Errorf("model and tokenizer are required")
	}
	sequence, err := prompt.Prepare(tok, state, q, model.Agent.MaxLen, model.Agent.HeadMaxLen)
	if err != nil {
		return Answer{}, prompt.Sequence{}, err
	}
	var qtype int
	switch q.Type {
	case "choice":
		qtype = 0
	case "score":
		qtype = 1
	case "noul":
		qtype = 2
	default:
		return Answer{}, prompt.Sequence{}, fmt.Errorf("unknown question type %q", q.Type)
	}
	logits, action, err := model.Forward(sequence.IDs, sequence.Markers, qtype)
	if err != nil {
		return Answer{}, prompt.Sequence{}, err
	}
	result, err := Format(q, logits, action, model.Agent)
	return result, sequence, err
}
