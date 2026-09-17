package mlx

import "testing"

func TestMetal(t *testing.T) {
	value, err := add(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if value != 3 {
		t.Fatalf("1 + 2 = %v, want 3", value)
	}
}
