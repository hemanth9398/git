package hasher

import (
	"encoding/binary"
	"hash/fnv"
	"strings"
)

const defaultMaxTokens = 256

// PrefixHasher computes a stable 64-bit hash over the first N whitespace-separated
// tokens of a prompt string. The hash is used for prefix-cache affinity routing:
// requests sharing the same system prompt or conversation context will hash to the
// same value and be routed to the node that already holds those KV-cache blocks.
type PrefixHasher struct {
	MaxTokens int
}

// New returns a PrefixHasher with the given token limit.
func New(maxTokens int) *PrefixHasher {
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	return &PrefixHasher{MaxTokens: maxTokens}
}

// Hash returns a 64-bit FNV-1a hash of the first MaxTokens tokens of text.
func (h *PrefixHasher) Hash(text string) uint64 {
	prefix := extractPrefix(text, h.MaxTokens)
	hasher := fnv.New64a()
	hasher.Write([]byte(prefix)) //nolint:errcheck
	return hasher.Sum64()
}

// HashTokenIDs returns a 64-bit hash over a slice of integer token IDs.
// This is the preferred path when the tokenizer output is already available.
func (h *PrefixHasher) HashTokenIDs(tokenIDs []int32) uint64 {
	n := len(tokenIDs)
	if n > h.MaxTokens {
		n = h.MaxTokens
	}
	hasher := fnv.New64a()
	buf := make([]byte, 4)
	for _, id := range tokenIDs[:n] {
		binary.LittleEndian.PutUint32(buf, uint32(id))
		hasher.Write(buf) //nolint:errcheck
	}
	return hasher.Sum64()
}

// extractPrefix returns the first n whitespace-separated tokens of text joined by spaces.
func extractPrefix(text string, n int) string {
	fields := strings.Fields(text)
	if len(fields) <= n {
		return text
	}
	return strings.Join(fields[:n], " ")
}
