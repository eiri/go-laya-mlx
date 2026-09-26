package checkpoint

import (
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const repo = "models--aac6fef--laya-mlx"

// Cached finds the local aac6fef/laya-mlx snapshot without network access.
func Cached() (string, error) {
	hub := os.Getenv("HF_HUB_CACHE")
	if hub == "" {
		home := os.Getenv("HF_HOME")
		if home == "" {
			base := os.Getenv("XDG_CACHE_HOME")
			if base == "" {
				userHome, err := os.UserHomeDir()
				if err != nil {
					return "", err
				}
				base = filepath.Join(userHome, ".cache")
			}
			home = filepath.Join(base, "huggingface")
		}
		hub = filepath.Join(home, "hub")
	}
	root := filepath.Join(hub, repo)

	// The Hub records the current revision in refs/main.
	ref, err := os.ReadFile(filepath.Join(root, "refs", "main"))
	if err != nil {
		return "", fmt.Errorf("model aac6fef/laya-mlx is not cached at %s: %w", root, err)
	}
	revision := strings.TrimSpace(string(ref))
	if len(revision) != 40 {
		return "", fmt.Errorf("invalid cached revision in %s", filepath.Join(root, "refs", "main"))
	}
	if _, err := hex.DecodeString(revision); err != nil {
		return "", fmt.Errorf("invalid cached revision %q: %w", revision, err)
	}
	dir := filepath.Join(root, "snapshots", revision)
	info, err := os.Stat(dir)
	if err != nil {
		return "", fmt.Errorf("cached model snapshot %s: %w", dir, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("cached model snapshot is not a directory: %s", dir)
	}
	return dir, nil
}
