package strategies

import "github.com/inference-gateway/routing"

// QueueDepthStrategy implements least-queue-depth load balancing. It scores
// nodes by their available queue headroom: a node with 0 waiting requests
// scores 1.0; a node at or above the saturation threshold scores near 0.
type QueueDepthStrategy struct {
	// SaturationDepth is the queue depth considered fully saturated. Defaults to 100.
	SaturationDepth float64
}

func (s *QueueDepthStrategy) Name() string { return "queue_depth" }

func (s *QueueDepthStrategy) Score(req routing.RoutingRequest, node routing.NodeInfo) float64 {
	saturation := s.SaturationDepth
	if saturation <= 0 {
		saturation = 100
	}
	waiting := float64(node.NumRequestsWaiting)
	if waiting >= saturation {
		return 0
	}
	return 1.0 - waiting/saturation
}
