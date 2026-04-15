package strategies

import "github.com/inference-gateway/routing"

// PrefixCacheAwareStrategy scores nodes based on whether they already hold the
// KV-cache blocks for the request's prompt prefix, enabling prefix-cache reuse
// and avoiding redundant prefill computation.
//
// Score logic:
//   - If the node has the exact prefix hash cached → 1.0 (full hit)
//   - Otherwise use the node's rolling prefix_cache_hit_rate as a proxy
type PrefixCacheAwareStrategy struct{}

func (s *PrefixCacheAwareStrategy) Name() string { return "prefix_cache_aware" }

func (s *PrefixCacheAwareStrategy) Score(req routing.RoutingRequest, node routing.NodeInfo) float64 {
	// Exact prefix hash hit: this node holds the KV-cache blocks for the prefix.
	if node.CachedPrefixHashes != nil && node.CachedPrefixHashes[req.PrefixHash] {
		return 1.0
	}
	// Fall back to the node's overall GPU cache hit rate as a soft affinity signal.
	if node.GPUCacheHitRate > 0 {
		return node.GPUCacheHitRate
	}
	return node.PrefixCacheHitRate
}
