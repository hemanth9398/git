// Package main implements the metrics aggregation server. It scrapes
// Prometheus metrics from all vLLM nodes and exposes an HTTP API that
// the ext_proc routing filter can query to make routing decisions.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

var (
	listenAddr  = flag.String("addr", ":9091", "HTTP listen address")
	scrapeInterval = flag.Duration("scrape-interval", 5*time.Second, "Metrics scrape interval")
	prefillAddrs   = flag.String("prefill-addrs", "", "Comma-separated prefill node metrics URLs")
	decodeAddrs    = flag.String("decode-addrs", "", "Comma-separated decode node metrics URLs")
)

// NodeMetrics holds scraped real-time metrics for a single vLLM node.
type NodeMetrics struct {
	NodeName            string             `json:"node_name"`
	Cluster             string             `json:"cluster"`
	KVCacheUsagePercent float64            `json:"kv_cache_usage_perc"`
	NumRequestsWaiting  int                `json:"num_requests_waiting"`
	PrefixCacheHitRate  float64            `json:"prefix_cache_hit_rate"`
	GPUCacheHitRate     float64            `json:"gpu_cache_hit_rate"`
	CachedPrefixHashes  map[uint64]bool    `json:"cached_prefix_hashes"`
	RequestSuccessTotal float64            `json:"request_success_total"`
	TTFT_P99            float64            `json:"ttft_p99_seconds"`
	ITL_P99             float64            `json:"itl_p99_seconds"`
	LastUpdated         time.Time          `json:"last_updated"`
	Healthy             bool               `json:"healthy"`
}

// Collector scrapes vLLM Prometheus metrics endpoints and aggregates results.
type Collector struct {
	log     *zap.Logger
	client  *http.Client

	mu    sync.RWMutex
	nodes map[string]*NodeMetrics

	// Prometheus gauges for the collector's own metrics.
	kvCacheGauge   *prometheus.GaugeVec
	queueGauge     *prometheus.GaugeVec
	prefixHitGauge *prometheus.GaugeVec
}

// NewCollector creates a metrics collector.
func NewCollector(log *zap.Logger) *Collector {
	kvCacheGauge := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "gateway_vllm_kv_cache_usage_perc",
		Help: "KV-cache usage percentage per vLLM node",
	}, []string{"node", "cluster"})

	queueGauge := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "gateway_vllm_num_requests_waiting",
		Help: "Number of requests waiting per vLLM node",
	}, []string{"node", "cluster"})

	prefixHitGauge := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "gateway_vllm_prefix_cache_hit_rate",
		Help: "Prefix cache hit rate per vLLM node",
	}, []string{"node", "cluster"})

	prometheus.MustRegister(kvCacheGauge, queueGauge, prefixHitGauge)

	return &Collector{
		log:            log,
		client:         &http.Client{Timeout: 3 * time.Second},
		nodes:          make(map[string]*NodeMetrics),
		kvCacheGauge:   kvCacheGauge,
		queueGauge:     queueGauge,
		prefixHitGauge: prefixHitGauge,
	}
}

// Run periodically scrapes all configured nodes until ctx is cancelled.
func (c *Collector) Run(ctx context.Context, prefillURLs, decodeURLs []string) {
	ticker := time.NewTicker(*scrapeInterval)
	defer ticker.Stop()

	// Initial scrape.
	c.scrapeAll(ctx, prefillURLs, decodeURLs)

	for {
		select {
		case <-ticker.C:
			c.scrapeAll(ctx, prefillURLs, decodeURLs)
		case <-ctx.Done():
			return
		}
	}
}

// scrapeAll concurrently scrapes all nodes.
func (c *Collector) scrapeAll(ctx context.Context, prefillURLs, decodeURLs []string) {
	type job struct {
		url     string
		cluster string
	}
	var jobs []job
	for _, u := range prefillURLs {
		jobs = append(jobs, job{url: u, cluster: "prefill"})
	}
	for _, u := range decodeURLs {
		jobs = append(jobs, job{url: u, cluster: "decode"})
	}

	var wg sync.WaitGroup
	for _, j := range jobs {
		j := j
		wg.Add(1)
		go func() {
			defer wg.Done()
			nm, err := c.scrapeNode(ctx, j.url, j.cluster)
			c.mu.Lock()
			if err != nil {
				if existing, ok := c.nodes[j.url]; ok {
					existing.Healthy = false
				}
			} else {
				c.nodes[j.url] = nm
				c.kvCacheGauge.WithLabelValues(nm.NodeName, nm.Cluster).Set(nm.KVCacheUsagePercent)
				c.queueGauge.WithLabelValues(nm.NodeName, nm.Cluster).Set(float64(nm.NumRequestsWaiting))
				c.prefixHitGauge.WithLabelValues(nm.NodeName, nm.Cluster).Set(nm.PrefixCacheHitRate)
			}
			c.mu.Unlock()
		}()
	}
	wg.Wait()
}

// scrapeNode fetches the /metrics endpoint of a vLLM node and parses key metrics.
func (c *Collector) scrapeNode(ctx context.Context, metricsURL, cluster string) (*NodeMetrics, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("scrape %s: %w", metricsURL, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}

	nm := &NodeMetrics{
		NodeName:    nodeNameFromURL(metricsURL),
		Cluster:     cluster,
		LastUpdated: time.Now(),
		Healthy:     true,
	}

	// Parse Prometheus text format (simple line scanner).
	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		parseMetricLine(line, nm)
	}
	return nm, nil
}

// parseMetricLine extracts known vLLM metrics from a Prometheus text line.
func parseMetricLine(line string, nm *NodeMetrics) {
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return
	}
	name := parts[0]
	var val float64
	fmt.Sscanf(parts[len(parts)-1], "%f", &val)

	switch {
	case strings.HasPrefix(name, "vllm:gpu_cache_usage_perc"):
		nm.KVCacheUsagePercent = val
	case strings.HasPrefix(name, "vllm:num_requests_waiting"):
		nm.NumRequestsWaiting = int(val)
	case strings.HasPrefix(name, "vllm:gpu_cache_hit_rate"):
		nm.GPUCacheHitRate = val
		nm.PrefixCacheHitRate = val
	case strings.HasPrefix(name, "vllm:request_success_total"):
		nm.RequestSuccessTotal = val
	}
}

// GetAllNodes returns a snapshot of current node metrics.
func (c *Collector) GetAllNodes() []NodeMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]NodeMetrics, 0, len(c.nodes))
	for _, nm := range c.nodes {
		out = append(out, *nm)
	}
	return out
}

func nodeNameFromURL(u string) string {
	u = strings.TrimPrefix(u, "http://")
	u = strings.TrimPrefix(u, "https://")
	if idx := strings.IndexByte(u, '/'); idx != -1 {
		u = u[:idx]
	}
	return u
}

func main() {
	flag.Parse()
	log, _ := zap.NewProduction()
	defer log.Sync() //nolint:errcheck

	collector := NewCollector(log)

	var prefillURLs, decodeURLs []string
	if *prefillAddrs != "" {
		for _, a := range strings.Split(*prefillAddrs, ",") {
			prefillURLs = append(prefillURLs, strings.TrimSpace(a))
		}
	}
	if *decodeAddrs != "" {
		for _, a := range strings.Split(*decodeAddrs, ",") {
			decodeURLs = append(decodeURLs, strings.TrimSpace(a))
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go collector.Run(ctx, prefillURLs, decodeURLs)

	mux := http.NewServeMux()

	// Node metrics API — consumed by ext_proc routing filter.
	mux.HandleFunc("/api/v1/node-metrics", func(w http.ResponseWriter, r *http.Request) {
		nodes := collector.GetAllNodes()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"nodes": nodes}) //nolint:errcheck
	})

	// Prometheus metrics endpoint (for Grafana).
	mux.Handle("/metrics", promhttp.Handler())

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`)) //nolint:errcheck
	})

	srv := &http.Server{
		Addr:         *listenAddr,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	log.Info("Metrics collector starting", zap.String("addr", *listenAddr))
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("metrics server error", zap.Error(err))
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	shutCtx, shutCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutCancel()
	srv.Shutdown(shutCtx) //nolint:errcheck
}
