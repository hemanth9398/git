package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"go.uber.org/zap"
)

// NodeMetrics holds real-time metrics for a single vLLM inference node.
type NodeMetrics struct {
	// NodeName is the Kubernetes pod name.
	NodeName string `json:"node_name"`
	// Cluster is "prefill" or "decode".
	Cluster string `json:"cluster"`
	// KVCacheUsagePercent is the fraction of GPU KV-cache blocks currently in use (0–100).
	KVCacheUsagePercent float64 `json:"kv_cache_usage_perc"`
	// NumRequestsWaiting is the number of requests currently queued (not running).
	NumRequestsWaiting int `json:"num_requests_waiting"`
	// PrefixCacheHitRate is the rolling hit rate of the radix-tree prefix cache (0–1).
	PrefixCacheHitRate float64 `json:"prefix_cache_hit_rate"`
	// GPUCacheHitRate is vLLM's native gpu_cache_hit_rate metric (0–1).
	GPUCacheHitRate float64 `json:"gpu_cache_hit_rate"`
	// CachedPrefixHashes is the set of prefix hashes currently resident on this node.
	CachedPrefixHashes map[uint64]bool `json:"cached_prefix_hashes"`
	// LastUpdated is the timestamp of the last successful scrape.
	LastUpdated time.Time `json:"last_updated"`
}

// MetricsClient scrapes the metrics aggregation API and caches results.
type MetricsClient struct {
	addr   string
	log    *zap.Logger
	client *http.Client

	mu      sync.RWMutex
	cached  []NodeMetrics
	cacheTS time.Time
	cacheTTL time.Duration
}

// NewMetricsClient constructs a MetricsClient pointing at the metrics aggregator API.
func NewMetricsClient(addr string, log *zap.Logger) *MetricsClient {
	return &MetricsClient{
		addr: addr,
		log:  log,
		client: &http.Client{
			Timeout: 200 * time.Millisecond,
		},
		cacheTTL: 500 * time.Millisecond,
	}
}

// GetAllNodeMetrics returns metrics for all vLLM nodes, using a short-lived cache
// to avoid hammering the metrics API on every request.
func (mc *MetricsClient) GetAllNodeMetrics(ctx context.Context) ([]NodeMetrics, error) {
	mc.mu.RLock()
	if time.Since(mc.cacheTS) < mc.cacheTTL && len(mc.cached) > 0 {
		cached := mc.cached
		mc.mu.RUnlock()
		return cached, nil
	}
	mc.mu.RUnlock()

	metrics, err := mc.fetchMetrics(ctx)
	if err != nil {
		// Return stale cache on error rather than failing the request.
		mc.mu.RLock()
		cached := mc.cached
		mc.mu.RUnlock()
		if len(cached) > 0 {
			mc.log.Warn("metrics fetch failed, using stale cache", zap.Error(err))
			return cached, nil
		}
		return nil, err
	}

	mc.mu.Lock()
	mc.cached = metrics
	mc.cacheTS = time.Now()
	mc.mu.Unlock()

	return metrics, nil
}

// fetchMetrics calls the metrics aggregation API and parses the JSON response.
func (mc *MetricsClient) fetchMetrics(ctx context.Context) ([]NodeMetrics, error) {
	url := fmt.Sprintf("http://%s/api/v1/node-metrics", mc.addr)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}

	resp, err := mc.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics API returned %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	var result struct {
		Nodes []NodeMetrics `json:"nodes"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}

	return result.Nodes, nil
}
