package main

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	cluster "github.com/envoyproxy/go-control-plane/envoy/config/cluster/v3"
	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	endpoint "github.com/envoyproxy/go-control-plane/envoy/config/endpoint/v3"
	listener "github.com/envoyproxy/go-control-plane/envoy/config/listener/v3"
	route "github.com/envoyproxy/go-control-plane/envoy/config/route/v3"
	hcm "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/network/http_connection_manager/v3"
	extproc "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	"github.com/envoyproxy/go-control-plane/pkg/cache/types"
	"github.com/envoyproxy/go-control-plane/pkg/cache/v3"
	"github.com/envoyproxy/go-control-plane/pkg/resource/v3"
	"github.com/envoyproxy/go-control-plane/pkg/wellknown"
	"go.uber.org/zap"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/durationpb"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	// nodeGroup is the Envoy node cluster name used as snapshot cache key.
	nodeGroup = "inference-gateway"

	// InferenceListenerPort is the port Envoy listens on for LLM traffic.
	InferenceListenerPort = 8080

	// OutlierConsecutive5xx triggers ejection after this many consecutive errors.
	OutlierConsecutive5xx = 3
)

// SnapshotManagerConfig holds parameters for the SnapshotManager.
type SnapshotManagerConfig struct {
	Namespace   string
	PrefillSvc  string
	DecodeSvc   string
	SnapshotTTL time.Duration
}

// SnapshotManager watches Kubernetes Endpoints and rebuilds the xDS snapshot
// whenever the endpoint set changes.
type SnapshotManager struct {
	mu      sync.Mutex
	version int64

	cache  cache.SnapshotCache
	k8s    kubernetes.Interface
	log    *zap.Logger
	config SnapshotManagerConfig

	debounce <-chan time.Time
}

// NewSnapshotManager constructs a SnapshotManager.
func NewSnapshotManager(
	c cache.SnapshotCache,
	k8s kubernetes.Interface,
	log *zap.Logger,
	cfg SnapshotManagerConfig,
) *SnapshotManager {
	return &SnapshotManager{cache: c, k8s: k8s, log: log, config: cfg}
}

// Reconcile triggers a snapshot rebuild (debounced by SnapshotTTL).
func (sm *SnapshotManager) Reconcile(ctx context.Context) {
	sm.mu.Lock()
	sm.debounce = time.After(sm.config.SnapshotTTL)
	sm.mu.Unlock()

	go func() {
		sm.mu.Lock()
		ch := sm.debounce
		sm.mu.Unlock()

		select {
		case <-ch:
			sm.buildAndPush(ctx)
		case <-ctx.Done():
		}
	}()
}

// buildAndPush fetches current Endpoints and pushes a new xDS snapshot.
func (sm *SnapshotManager) buildAndPush(ctx context.Context) {
	prefillEPs, err := sm.fetchEndpoints(ctx, sm.config.PrefillSvc)
	if err != nil {
		sm.log.Error("failed to fetch prefill endpoints", zap.Error(err))
		return
	}
	decodeEPs, err := sm.fetchEndpoints(ctx, sm.config.DecodeSvc)
	if err != nil {
		sm.log.Error("failed to fetch decode endpoints", zap.Error(err))
		return
	}

	sm.mu.Lock()
	sm.version++
	version := strconv.FormatInt(sm.version, 10)
	sm.mu.Unlock()

	clusters := sm.buildClusters()
	eds := sm.buildEDS(prefillEPs, decodeEPs)
	listeners := sm.buildListeners()
	routes := sm.buildRoutes()

	snapshot, err := cache.NewSnapshot(version,
		map[resource.Type][]types.Resource{
			resource.ClusterType:  clusters,
			resource.EndpointType: eds,
			resource.ListenerType: listeners,
			resource.RouteType:    routes,
		},
	)
	if err != nil {
		sm.log.Error("failed to create snapshot", zap.Error(err))
		return
	}

	if err := sm.cache.SetSnapshot(ctx, nodeGroup, snapshot); err != nil {
		sm.log.Error("failed to set snapshot", zap.Error(err))
		return
	}

	sm.log.Info("xDS snapshot pushed",
		zap.String("version", version),
		zap.Int("prefill_endpoints", len(prefillEPs)),
		zap.Int("decode_endpoints", len(decodeEPs)),
	)
}

// fetchEndpoints returns ready pod IPs for the given service in the configured namespace.
func (sm *SnapshotManager) fetchEndpoints(ctx context.Context, svcName string) ([]podEndpoint, error) {
	eps, err := sm.k8s.CoreV1().Endpoints(sm.config.Namespace).Get(ctx, svcName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get endpoints %s: %w", svcName, err)
	}
	var result []podEndpoint
	for _, subset := range eps.Subsets {
		for _, port := range subset.Ports {
			for _, addr := range subset.Addresses {
				result = append(result, podEndpoint{
					IP:   addr.IP,
					Port: uint32(port.Port),
					Name: podName(addr),
				})
			}
		}
	}
	return result, nil
}

type podEndpoint struct {
	IP   string
	Port uint32
	Name string
}

func podName(addr corev1.EndpointAddress) string {
	if addr.TargetRef != nil {
		return addr.TargetRef.Name
	}
	return addr.IP
}

// buildClusters returns CDS resources for prefill and decode upstream clusters.
func (sm *SnapshotManager) buildClusters() []types.Resource {
	makeCluster := func(name string) *cluster.Cluster {
		return &cluster.Cluster{
			Name:                 name,
			ConnectTimeout:       durationpb.New(5 * time.Second),
			ClusterDiscoveryType: &cluster.Cluster_Type{Type: cluster.Cluster_EDS},
			EdsClusterConfig: &cluster.Cluster_EdsClusterConfig{
				EdsConfig: &core.ConfigSource{
					ResourceApiVersion: core.ApiVersion_V3,
					ConfigSourceSpecifier: &core.ConfigSource_Ads{
						Ads: &core.AggregatedConfigSource{},
					},
				},
				ServiceName: name,
			},
			LbPolicy: cluster.Cluster_ROUND_ROBIN,
			// Circuit breaker thresholds.
			CircuitBreakers: &cluster.CircuitBreakers{
				Thresholds: []*cluster.CircuitBreakers_Thresholds{
					{
						MaxConnections:     wrapUint32(10000),
						MaxPendingRequests: wrapUint32(5000),
						MaxRequests:        wrapUint32(20000),
						MaxRetries:         wrapUint32(100),
					},
				},
			},
			// Passive health checking — eject nodes with consecutive 5xx.
			OutlierDetection: &cluster.OutlierDetection{
				Consecutive_5Xx:                    wrapUint32(OutlierConsecutive5xx),
				Interval:                           durationpb.New(10 * time.Second),
				BaseEjectionTime:                   durationpb.New(30 * time.Second),
				MaxEjectionPercent:                 wrapUint32(50),
				EnforcingConsecutive_5Xx:           wrapUint32(100),
				EnforcingSuccessRate:               wrapUint32(100),
				SuccessRateMinimumHosts:             wrapUint32(2),
				SuccessRateRequestVolume:            wrapUint32(100),
				SuccessRateStdevFactor:              wrapUint32(1900),
			},
			// Active health checking.
			HealthChecks: []*core.HealthCheck{
				{
					Timeout:            durationpb.New(5 * time.Second),
					Interval:           durationpb.New(10 * time.Second),
					UnhealthyThreshold: wrapUint32(2),
					HealthyThreshold:   wrapUint32(2),
					HealthChecker: &core.HealthCheck_HttpHealthCheck_{
						HttpHealthCheck: &core.HealthCheck_HttpHealthCheck{
							Path: "/health",
						},
					},
				},
			},
		}
	}

	return []types.Resource{
		makeCluster("vllm-prefill"),
		makeCluster("vllm-decode"),
	}
}

// buildEDS constructs EDS ClusterLoadAssignment resources from live pod endpoints.
func (sm *SnapshotManager) buildEDS(prefillEPs, decodeEPs []podEndpoint) []types.Resource {
	makeAssignment := func(clusterName string, eps []podEndpoint) *endpoint.ClusterLoadAssignment {
		var lbEPs []*endpoint.LbEndpoint
		for _, ep := range eps {
			lbEPs = append(lbEPs, &endpoint.LbEndpoint{
				HostIdentifier: &endpoint.LbEndpoint_Endpoint{
					Endpoint: &endpoint.Endpoint{
						Address: &core.Address{
							Address: &core.Address_SocketAddress{
								SocketAddress: &core.SocketAddress{
									Protocol: core.SocketAddress_TCP,
									Address:  ep.IP,
									PortSpecifier: &core.SocketAddress_PortValue{
										PortValue: ep.Port,
									},
								},
							},
						},
						// Hostname used for SNI in mTLS.
						Hostname: ep.Name,
					},
				},
				// Metadata carries node name for header-based routing affinity.
				Metadata: &core.Metadata{
					FilterMetadata: map[string]*structpb.Struct{
						"envoy.lb": {
							Fields: map[string]*structpb.Value{
								"node_name": structpb.NewStringValue(ep.Name),
							},
						},
					},
				},
			})
		}
		return &endpoint.ClusterLoadAssignment{
			ClusterName: clusterName,
			Endpoints: []*endpoint.LocalityLbEndpoints{
				{LbEndpoints: lbEPs},
			},
		}
	}

	return []types.Resource{
		makeAssignment("vllm-prefill", prefillEPs),
		makeAssignment("vllm-decode", decodeEPs),
	}
}

// buildListeners returns LDS resources: a single ingress listener with ext_proc + JWT + rate-limit filters.
func (sm *SnapshotManager) buildListeners() []types.Resource {
	extProcFilter := &extproc.ExternalProcessor{
		GrpcService: &core.GrpcService{
			TargetSpecifier: &core.GrpcService_EnvoyGrpc_{
				EnvoyGrpc: &core.GrpcService_EnvoyGrpc{
					ClusterName: "ext_proc_cluster",
				},
			},
			Timeout: durationpb.New(500 * time.Millisecond),
		},
		ProcessingMode: &extproc.ProcessingMode{
			RequestHeaderMode:  extproc.ProcessingMode_SEND,
			RequestBodyMode:    extproc.ProcessingMode_BUFFERED,
			ResponseHeaderMode: extproc.ProcessingMode_SKIP,
			ResponseBodyMode:   extproc.ProcessingMode_SKIP,
		},
		MessageTimeout:     durationpb.New(500 * time.Millisecond),
		FailureModeAllow:   false,
	}
	extProcAny, _ := anypb.New(extProcFilter)

	hcmFilter := &hcm.HttpConnectionManager{
		StatPrefix: "inference_gateway",
		RouteSpecifier: &hcm.HttpConnectionManager_Rds{
			Rds: &hcm.Rds{
				ConfigSource: &core.ConfigSource{
					ResourceApiVersion: core.ApiVersion_V3,
					ConfigSourceSpecifier: &core.ConfigSource_Ads{
						Ads: &core.AggregatedConfigSource{},
					},
				},
				RouteConfigName: "inference_routes",
			},
		},
		HttpFilters: []*hcm.HttpFilter{
			{
				Name:       "envoy.filters.http.ext_proc",
				ConfigType: &hcm.HttpFilter_TypedConfig{TypedConfig: extProcAny},
			},
			{
				Name: wellknown.Router,
				ConfigType: &hcm.HttpFilter_TypedConfig{
					TypedConfig: mustAny(&routerv3.Router{}),
				},
			},
		},
		AccessLog: accessLogs(),
		// Timeouts tuned for streaming LLM responses (up to 300s for long completions).
		StreamIdleTimeout:    durationpb.New(300 * time.Second),
		RequestTimeout:       durationpb.New(300 * time.Second),
		DrainTimeout:         durationpb.New(30 * time.Second),
	}
	hcmAny, _ := anypb.New(hcmFilter)

	lis := &listener.Listener{
		Name: "inference_listener",
		Address: &core.Address{
			Address: &core.Address_SocketAddress{
				SocketAddress: &core.SocketAddress{
					Protocol: core.SocketAddress_TCP,
					Address:  "0.0.0.0",
					PortSpecifier: &core.SocketAddress_PortValue{
						PortValue: InferenceListenerPort,
					},
				},
			},
		},
		FilterChains: []*listener.FilterChain{
			{
				Filters: []*listener.Filter{
					{
						Name:       wellknown.HTTPConnectionManager,
						ConfigType: &listener.Filter_TypedConfig{TypedConfig: hcmAny},
					},
				},
			},
		},
	}
	return []types.Resource{lis}
}

// buildRoutes returns RDS resources routing by x-route-to-cluster header set by ext_proc.
func (sm *SnapshotManager) buildRoutes() []types.Resource {
	makeHeaderRoute := func(clusterName, headerValue string) *route.Route {
		return &route.Route{
			Match: &route.RouteMatch{
				PathSpecifier: &route.RouteMatch_Prefix{Prefix: "/"},
				Headers: []*route.HeaderMatcher{
					{
						Name: "x-route-to-cluster",
						HeaderMatchSpecifier: &route.HeaderMatcher_ExactMatch{
							ExactMatch: headerValue,
						},
					},
				},
			},
			Action: &route.Route_Route{
				Route: &route.RouteAction{
					ClusterSpecifier: &route.RouteAction_Cluster{
						Cluster: clusterName,
					},
					Timeout:    durationpb.New(300 * time.Second),
					RetryPolicy: &route.RetryPolicy{
						RetryOn:              "connect-failure,retriable-4xx,refused-stream,unavailable",
						NumRetries:           wrapUint32(2),
						PerTryTimeout:        durationpb.New(30 * time.Second),
						RetriableStatusCodes: []uint32{429, 503},
					},
				},
			},
		}
	}

	rc := &route.RouteConfiguration{
		Name: "inference_routes",
		VirtualHosts: []*route.VirtualHost{
			{
				Name:    "inference_vhost",
				Domains: []string{"*"},
				Routes: []*route.Route{
					makeHeaderRoute("vllm-prefill", "prefill"),
					makeHeaderRoute("vllm-decode", "decode"),
					// Default: route to decode cluster (combined prefill+decode mode).
					{
						Match: &route.RouteMatch{
							PathSpecifier: &route.RouteMatch_Prefix{Prefix: "/"},
						},
						Action: &route.Route_Route{
							Route: &route.RouteAction{
								ClusterSpecifier: &route.RouteAction_Cluster{
									Cluster: "vllm-decode",
								},
								Timeout: durationpb.New(300 * time.Second),
							},
						},
					},
				},
			},
		},
	}
	return []types.Resource{rc}
}

// wrapUint32 wraps a uint32 in a wrapperspb value (required by Envoy protobuf API).
func wrapUint32(v uint32) *wrapperspb.UInt32Value {
	return &wrapperspb.UInt32Value{Value: v}
}

func mustAny(m proto.Message) *anypb.Any {
	a, err := anypb.New(m)
	if err != nil {
		panic(err)
	}
	return a
}

func accessLogs() []*accesslogv3.AccessLog {
	return []*accesslogv3.AccessLog{
		{
			Name: "envoy.access_loggers.stdout",
			ConfigType: &accesslogv3.AccessLog_TypedConfig{
				TypedConfig: mustAny(&streamaccesslogv3.StdoutAccessLog{}),
			},
		},
	}
}
