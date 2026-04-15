package strategies

import "github.com/inference-gateway/routing"

// KVCacheAwareStrategy routes away from nodes whose GPU KV-cache is heavily
// utilized. A node at 0% usage scores 1.0; a node at 100% scores 0.0.
// Nodes above the eviction threshold (default 85%) receive a steep penalty.
type KVCacheAwareStrategy struct {
	// EvictionThreshold is the KV-cache % above which the penalty steepens.
	// Defaults to 85.
	EvictionThreshold float64
}

func (s *KVCacheAwareStrategy) Name() string { return "kv_cache_aware" }

func (s *KVCacheAwareStrategy) Score(req routing.RoutingRequest, node routing.NodeInfo) float64 {
	threshold := s.EvictionThreshold
	if threshold == 0 {
		threshold = 85
	}
	usage := node.KVCacheUsagePerc
	if usage >= 100 {
		return 0
	}
	headroom := 1.0 - usage/100.0

	// Apply a steeper penalty once the node exceeds the eviction threshold.
	if usage > threshold {
		// Linear penalty: at threshold → headroom; at 100% → 0.
		// Multiply headroom by a shrinkage factor so over-threshold nodes
		// are deprioritised more aggressively.
		excess := (usage - threshold) / (100 - threshold) // 0..1
		headroom *= (1 - 0.8*excess)
	}
	return headroom
}
