// Package main implements the ext_proc gRPC server that Envoy calls for every
// inference request. It computes a routing score across available vLLM nodes
// using real-time KV-cache utilization, queue depth, and prefix-cache hit-rate
// metrics, then injects the x-route-to-cluster header for Envoy's header-based
// routing to select the optimal upstream.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extprocv3 "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
)

var (
	listenAddr     = flag.String("addr", ":9090", "ext_proc gRPC listen address")
	metricsAddr    = flag.String("metrics-addr", "metrics-server.inference-gateway.svc.cluster.local:9091", "Metrics API address")
	prefixTokens   = flag.Int("prefix-tokens", 256, "Number of prefix tokens to hash for cache affinity")
	alphaWeight    = flag.Float64("alpha", 0.40, "Weight for prefix-cache hit rate in routing score")
	betaWeight     = flag.Float64("beta", 0.35, "Weight for KV-cache headroom in routing score")
	gammaWeight    = flag.Float64("gamma", 0.25, "Weight for queue-depth headroom in routing score")
)

// processorServer implements the ExternalProcessor gRPC service.
type processorServer struct {
	extprocv3.UnimplementedExternalProcessorServer
	metrics *MetricsClient
	log     *zap.Logger
	cfg     processorConfig
}

type processorConfig struct {
	PrefixTokens int
	Alpha        float64
	Beta         float64
	Gamma        float64
}

// Process handles the bidirectional streaming RPC from Envoy's ext_proc filter.
// For each HTTP request Envoy sends us the request headers and optionally the
// body; we return a header mutation with the routing decision.
func (s *processorServer) Process(stream extprocv3.ExternalProcessor_ProcessServer) error {
	ctx := stream.Context()
	startTime := time.Now()

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			if status.Code(err) == codes.Canceled {
				return nil
			}
			s.log.Error("stream recv error", zap.Error(err))
			return err
		}

		var resp *extprocv3.ProcessingResponse

		switch msg := req.Request.(type) {
		case *extprocv3.ProcessingRequest_RequestHeaders:
			resp = s.handleRequestHeaders(ctx, msg.RequestHeaders)

		case *extprocv3.ProcessingRequest_RequestBody:
			resp = s.handleRequestBody(ctx, msg.RequestBody)

		default:
			resp = &extprocv3.ProcessingResponse{
				Response: &extprocv3.ProcessingResponse_RequestHeaders{
					RequestHeaders: &extprocv3.HeadersResponse{},
				},
			}
		}

		if err := stream.Send(resp); err != nil {
			s.log.Error("stream send error", zap.Error(err))
			return err
		}
	}

	_ = startTime
	return nil
}

// handleRequestHeaders extracts the authorization header and returns immediately
// (body buffering is configured in Envoy, so we wait for handleRequestBody).
func (s *processorServer) handleRequestHeaders(ctx context.Context, hdrs *extprocv3.HttpHeaders) *extprocv3.ProcessingResponse {
	return &extprocv3.ProcessingResponse{
		Response: &extprocv3.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extprocv3.HeadersResponse{
				Response: &extprocv3.CommonResponse{
					Status: extprocv3.CommonResponse_CONTINUE,
				},
			},
		},
	}
}

// handleRequestBody reads the buffered request body, extracts the prompt prefix,
// computes a routing score, and injects the x-route-to-cluster header.
func (s *processorServer) handleRequestBody(ctx context.Context, body *extprocv3.HttpBody) *extprocv3.ProcessingResponse {
	cluster, score := s.routingDecision(ctx, body.Body)

	s.log.Debug("routing decision",
		zap.String("cluster", cluster),
		zap.Float64("score", score),
	)

	return &extprocv3.ProcessingResponse{
		Response: &extprocv3.ProcessingResponse_RequestBody{
			RequestBody: &extprocv3.BodyResponse{
				Response: &extprocv3.CommonResponse{
					Status: extprocv3.CommonResponse_CONTINUE,
					HeaderMutation: &extprocv3.HeaderMutation{
						SetHeaders: []*corev3.HeaderValueOption{
							{
								Header: &corev3.HeaderValue{
									Key:   "x-route-to-cluster",
									Value: cluster,
								},
							},
						},
					},
				},
			},
		},
	}
}

// routingDecision computes a composite routing score for each available node
// and returns the cluster name of the best node.
func (s *processorServer) routingDecision(ctx context.Context, body []byte) (string, float64) {
	// Parse prompt from OpenAI-compatible request body.
	prefix := extractPromptPrefix(body, s.cfg.PrefixTokens)
	prefixHash := hashPrefix(prefix)

	// Fetch live metrics for all nodes.
	nodeMetrics, err := s.metrics.GetAllNodeMetrics(ctx)
	if err != nil || len(nodeMetrics) == 0 {
		// Fallback: default to decode cluster.
		return "decode", 0
	}

	// Score each node.
	type scored struct {
		cluster string
		score   float64
	}
	var best scored
	for _, nm := range nodeMetrics {
		prefixHit := 0.0
		if nm.PrefixCacheHitRate > 0 {
			prefixHit = nm.PrefixCacheHitRate
		}
		// Bonus: if this node has the prefix cached, boost by affinity.
		if nm.CachedPrefixHashes[prefixHash] {
			prefixHit = 1.0
		}

		kvHeadroom := 1.0 - nm.KVCacheUsagePercent/100.0
		queueHeadroom := 1.0 - min(float64(nm.NumRequestsWaiting)/100.0, 1.0)

		score := s.cfg.Alpha*prefixHit + s.cfg.Beta*kvHeadroom + s.cfg.Gamma*queueHeadroom

		if score > best.score {
			best = scored{cluster: nm.Cluster, score: score}
		}
	}

	if best.cluster == "" {
		return "decode", 0
	}
	return best.cluster, best.score
}

// extractPromptPrefix parses an OpenAI-compatible JSON body and returns
// the first maxTokens whitespace-separated tokens of the concatenated messages.
func extractPromptPrefix(body []byte, maxTokens int) string {
	var req struct {
		Prompt   string `json:"prompt"`
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	text := req.Prompt
	for _, m := range req.Messages {
		text += " " + m.Content
	}
	// Simple whitespace tokenization (real implementation would use tiktoken/sentencepiece).
	tokens := splitTokens(text, maxTokens)
	return tokens
}

// splitTokens returns at most n space-separated tokens joined back as a string.
func splitTokens(text string, n int) string {
	words := make([]byte, 0, len(text))
	count := 0
	inWord := false
	for i := 0; i < len(text) && count < n; i++ {
		c := text[i]
		if c == ' ' || c == '\t' || c == '\n' {
			if inWord {
				count++
				inWord = false
			}
			if count < n {
				words = append(words, c)
			}
		} else {
			inWord = true
			words = append(words, c)
		}
	}
	return string(words)
}

// hashPrefix returns a 64-bit FNV-1a hash of the prefix string.
func hashPrefix(prefix string) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	for i := 0; i < len(prefix); i++ {
		h ^= uint64(prefix[i])
		h *= prime64
	}
	return h
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func main() {
	flag.Parse()
	log, _ := zap.NewProduction()
	defer log.Sync() //nolint:errcheck

	mc := NewMetricsClient(*metricsAddr, log)

	srv := &processorServer{
		metrics: mc,
		log:     log,
		cfg: processorConfig{
			PrefixTokens: *prefixTokens,
			Alpha:        *alphaWeight,
			Beta:         *betaWeight,
			Gamma:        *gammaWeight,
		},
	}

	grpcSrv := grpc.NewServer(
		grpc.MaxConcurrentStreams(10000),
		grpc.KeepaliveParams(keepalive.ServerParameters{
			Time:    30 * time.Second,
			Timeout: 10 * time.Second,
		}),
	)
	extprocv3.RegisterExternalProcessorServer(grpcSrv, srv)

	lis, err := net.Listen("tcp", *listenAddr)
	if err != nil {
		log.Fatal("failed to listen", zap.Error(err))
	}
	log.Info("ext_proc server starting", zap.String("addr", *listenAddr))

	go func() {
		if err := grpcSrv.Serve(lis); err != nil {
			fmt.Fprintf(os.Stderr, "ext_proc server error: %v\n", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	grpcSrv.GracefulStop()
}
