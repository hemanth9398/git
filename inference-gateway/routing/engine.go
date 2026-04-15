// Package routing implements the core routing decision engine for the
// LLM inference gateway. It combines multiple routing strategies into a
// composite score to select the optimal vLLM node for each request.
package routing

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"
)

// NodeInfo holds real-time state for a single vLLM backend node.
type NodeInfo struct {
	Name               string
	Cluster            string // "prefill" or "decode"
	KVCacheUsagePerc   float64
	NumRequestsWaiting int
	PrefixCacheHitRate float64
	GPUCacheHitRate    float64
	CachedPrefixHashes map[uint64]bool
	LastSeen           time.Time
}

// RoutingRequest carries the context needed to make a routing decision.
type RoutingRequest struct {
	// PrefixHash is the FNV-1a hash of the prompt prefix tokens.
	PrefixHash uint64
	// PrefixText is the raw prefix text (first N tokens).
	PrefixText string
	// TenantID is used for tenant-level priority routing.
	TenantID string
	// RequestedCluster forces routing to a specific cluster ("prefill"/"decode"/"").
	RequestedCluster string
}

// RoutingDecision is the output of the routing engine.
type RoutingDecision struct {
	// TargetCluster is the Envoy cluster name to route to.
	TargetCluster string
	// TargetNode is the preferred backend pod name (used as sticky hint).
	TargetNode string
	// Score is the composite routing score (higher is better).
	Score float64
	// Strategy is the name of the strategy that produced this decision.
	Strategy string
}

// Strategy is implemented by each routing algorithm.
type Strategy interface {
	Name() string
	Score(req RoutingRequest, node NodeInfo) float64
}

// NodeProvider returns the current set of available nodes.
type NodeProvider interface {
	GetNodes(ctx context.Context) ([]NodeInfo, error)
}

// Engine is the top-level routing decision engine. It evaluates all registered
// strategies, combines their scores with configured weights, and returns the
// best node.
type Engine struct {
	provider   NodeProvider
	strategies []weightedStrategy
	log        *zap.Logger
}

type weightedStrategy struct {
	strategy Strategy
	weight   float64
}

// EngineConfig holds the configuration for the routing engine.
type EngineConfig struct {
	// Strategies maps strategy name → weight. Weights are normalized internally.
	Strategies map[string]float64
}

// NewEngine constructs a routing engine with the given node provider and config.
func NewEngine(provider NodeProvider, cfg EngineConfig, log *zap.Logger) (*Engine, error) {
	strategies := []Strategy{
		&strategies.KVCacheAwareStrategy{},
		&strategies.PrefixCacheAwareStrategy{},
		&strategies.QueueDepthStrategy{},
	}

	var ws []weightedStrategy
	totalWeight := 0.0
	for _, s := range strategies {
		w, ok := cfg.Strategies[s.Name()]
		if !ok {
			continue
		}
		if w <= 0 {
			continue
		}
		ws = append(ws, weightedStrategy{strategy: s, weight: w})
		totalWeight += w
	}
	if len(ws) == 0 {
		return nil, fmt.Errorf("no strategies configured")
	}
	// Normalize weights.
	for i := range ws {
		ws[i].weight /= totalWeight
	}

	return &Engine{provider: provider, strategies: ws, log: log}, nil
}

// Route evaluates all strategies against all available nodes and returns
// the best routing decision.
func (e *Engine) Route(ctx context.Context, req RoutingRequest) (RoutingDecision, error) {
	nodes, err := e.provider.GetNodes(ctx)
	if err != nil {
		return RoutingDecision{}, fmt.Errorf("get nodes: %w", err)
	}
	if len(nodes) == 0 {
		return RoutingDecision{TargetCluster: "vllm-decode"}, nil
	}

	// Filter to requested cluster if specified.
	if req.RequestedCluster != "" {
		filtered := nodes[:0]
		for _, n := range nodes {
			if n.Cluster == req.RequestedCluster {
				filtered = append(filtered, n)
			}
		}
		if len(filtered) > 0 {
			nodes = filtered
		}
	}

	type candidate struct {
		node  NodeInfo
		score float64
	}

	var best candidate
	for _, node := range nodes {
		// Skip nodes with KV cache nearly full.
		if node.KVCacheUsagePerc > 95 {
			e.log.Debug("skipping near-full node",
				zap.String("node", node.Name),
				zap.Float64("kv_usage", node.KVCacheUsagePerc),
			)
			continue
		}

		composite := 0.0
		for _, ws := range e.strategies {
			composite += ws.weight * ws.strategy.Score(req, node)
		}

		if composite > best.score {
			best = candidate{node: node, score: composite}
		}
	}

	if best.node.Name == "" {
		// All nodes were filtered; fall back to least-loaded.
		best = leastLoaded(nodes)
	}

	decision := RoutingDecision{
		TargetCluster: clusterName(best.node.Cluster),
		TargetNode:    best.node.Name,
		Score:         best.score,
		Strategy:      "composite",
	}
	e.log.Debug("routing decision",
		zap.String("cluster", decision.TargetCluster),
		zap.String("node", decision.TargetNode),
		zap.Float64("score", decision.Score),
	)
	return decision, nil
}

func leastLoaded(nodes []NodeInfo) struct {
	node  NodeInfo
	score float64
} {
	type c struct {
		node  NodeInfo
		score float64
	}
	best := c{}
	for _, n := range nodes {
		s := 1.0 - float64(n.NumRequestsWaiting)/100.0
		if s > best.score {
			best = c{node: n, score: s}
		}
	}
	return best
}

func clusterName(cluster string) string {
	switch cluster {
	case "prefill":
		return "vllm-prefill"
	case "decode":
		return "vllm-decode"
	default:
		return "vllm-decode"
	}
}
