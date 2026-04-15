package strategies

import (
	"github.com/inference-gateway/routing"
)

// CompositeStrategy combines multiple strategies with configured weights into a
// single score: score = Σ(weight_i * strategy_i.Score(req, node)).
// Weights are normalized to sum to 1.0.
type CompositeStrategy struct {
	components []component
}

type component struct {
	strategy routing.Strategy
	weight   float64
}

// NewCompositeStrategy constructs a CompositeStrategy from a map of
// strategy-name → weight. Unknown strategy names are ignored.
func NewCompositeStrategy(weights map[string]float64) *CompositeStrategy {
	allStrategies := []routing.Strategy{
		&KVCacheAwareStrategy{},
		&PrefixCacheAwareStrategy{},
		&QueueDepthStrategy{},
	}

	var components []component
	total := 0.0
	for _, s := range allStrategies {
		w, ok := weights[s.Name()]
		if !ok || w <= 0 {
			continue
		}
		components = append(components, component{strategy: s, weight: w})
		total += w
	}
	// Normalize.
	for i := range components {
		components[i].weight /= total
	}
	return &CompositeStrategy{components: components}
}

func (s *CompositeStrategy) Name() string { return "composite" }

func (s *CompositeStrategy) Score(req routing.RoutingRequest, node routing.NodeInfo) float64 {
	score := 0.0
	for _, c := range s.components {
		score += c.weight * c.strategy.Score(req, node)
	}
	return score
}
