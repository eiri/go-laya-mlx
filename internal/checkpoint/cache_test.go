package checkpoint

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCached(t *testing.T) {
	for _, tc := range []struct {
		name, env string
	}{
		{"hub", "HF_HUB_CACHE"},
		{"hf-home", "HF_HOME"},
		{"xdg", "XDG_CACHE_HOME"},
		{"default", "HOME"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := t.TempDir()
			for _, name := range []string{"HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "HOME"} {
				t.Setenv(name, "")
			}
			t.Setenv(tc.env, base)
			if tc.env == "HF_HUB_CACHE" {
				t.Setenv("HF_HOME", filepath.Join(base, "wrong-home"))
				t.Setenv("XDG_CACHE_HOME", filepath.Join(base, "wrong-xdg"))
			}
			var hub string
			switch tc.env {
			case "HF_HUB_CACHE":
				hub = base
			case "HF_HOME":
				hub = filepath.Join(base, "hub")
			case "XDG_CACHE_HOME":
				hub = filepath.Join(base, "huggingface", "hub")
			default:
				hub = filepath.Join(base, ".cache", "huggingface", "hub")
			}
			root := filepath.Join(hub, repo)
			revision := strings.Repeat("a", 40)
			want := filepath.Join(root, "snapshots", revision)
			if err := os.MkdirAll(filepath.Join(root, "refs"), 0755); err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(want, 0755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, "refs", "main"), []byte(revision+"\n"), 0644); err != nil {
				t.Fatal(err)
			}
			got, err := Cached()
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("got %q, want %q", got, want)
			}
		})
	}
}

func TestCacheMissing(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	if _, err := Cached(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing cache: %v", err)
	}
}

func TestCacheBadRevision(t *testing.T) {
	hub := t.TempDir()
	t.Setenv("HF_HUB_CACHE", hub)
	path := filepath.Join(hub, repo, "refs", "main")
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("../../outside"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := Cached(); err == nil {
		t.Fatal("accepted invalid ref")
	}
}
