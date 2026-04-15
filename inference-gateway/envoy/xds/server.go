// Package main implements a gRPC xDS control plane server (LDS/RDS/CDS/EDS/SDS)
// for the Envoy-based LLM inference gateway. It watches Kubernetes Endpoints
// and dynamically pushes cluster/endpoint/listener/route snapshots to Envoy
// proxies without requiring restarts.
package main

import (
	"context"
	"flag"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	clusterservice "github.com/envoyproxy/go-control-plane/envoy/service/cluster/v3"
	discoveryservice "github.com/envoyproxy/go-control-plane/envoy/service/discovery/v3"
	endpointservice "github.com/envoyproxy/go-control-plane/envoy/service/endpoint/v3"
	listenerservice "github.com/envoyproxy/go-control-plane/envoy/service/listener/v3"
	routeservice "github.com/envoyproxy/go-control-plane/envoy/service/route/v3"
	"github.com/envoyproxy/go-control-plane/pkg/cache/v3"
	"github.com/envoyproxy/go-control-plane/pkg/server/v3"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
)

var (
	xdsPort       = flag.Int("xds-port", 18000, "gRPC xDS server port")
	namespace     = flag.String("namespace", "inference-gateway", "Kubernetes namespace to watch")
	prefillSvc    = flag.String("prefill-service", "vllm-prefill", "Prefill service name")
	decodeSvc     = flag.String("decode-service", "vllm-decode", "Decode service name")
	snapshotTTL   = flag.Duration("snapshot-ttl", 100*time.Millisecond, "Max delay for snapshot push after endpoint change")
)

// xdsCallbacks implements server.Callbacks for logging xDS events.
type xdsCallbacks struct {
	log *zap.Logger
}

func (cb *xdsCallbacks) Report()                                                    {}
func (cb *xdsCallbacks) OnStreamOpen(_ context.Context, id int64, typ string) error {
	cb.log.Info("xDS stream opened", zap.Int64("id", id), zap.String("type", typ))
	return nil
}
func (cb *xdsCallbacks) OnStreamClosed(id int64, node *core.Node) {
	cb.log.Info("xDS stream closed", zap.Int64("id", id))
}
func (cb *xdsCallbacks) OnDeltaStreamOpen(_ context.Context, id int64, typ string) error {
	return nil
}
func (cb *xdsCallbacks) OnDeltaStreamClosed(id int64, node *core.Node) {}
func (cb *xdsCallbacks) OnStreamRequest(id int64, req *discoveryservice.DiscoveryRequest) error {
	return nil
}
func (cb *xdsCallbacks) OnStreamResponse(_ context.Context, id int64, req *discoveryservice.DiscoveryRequest, resp *discoveryservice.DiscoveryResponse) {
}
func (cb *xdsCallbacks) OnStreamDeltaRequest(id int64, req *discoveryservice.DeltaDiscoveryRequest) error {
	return nil
}
func (cb *xdsCallbacks) OnStreamDeltaResponse(id int64, req *discoveryservice.DeltaDiscoveryRequest, resp *discoveryservice.DeltaDiscoveryResponse) {
}
func (cb *xdsCallbacks) OnFetchRequest(_ context.Context, req *discoveryservice.DiscoveryRequest) error {
	return nil
}
func (cb *xdsCallbacks) OnFetchResponse(req *discoveryservice.DiscoveryRequest, resp *discoveryservice.DiscoveryResponse) {
}

func main() {
	flag.Parse()

	log, _ := zap.NewProduction()
	defer log.Sync() //nolint:errcheck

	// Build in-cluster Kubernetes client.
	cfg, err := rest.InClusterConfig()
	if err != nil {
		log.Fatal("failed to build in-cluster config", zap.Error(err))
	}
	k8sClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		log.Fatal("failed to create k8s client", zap.Error(err))
	}

	// Create xDS snapshot cache (one snapshot per Envoy node group).
	snapshotCache := cache.NewSnapshotCache(false, cache.IDHash{}, nil)

	// Build snapshot manager — watches K8s Endpoints and pushes snapshots.
	sm := NewSnapshotManager(snapshotCache, k8sClient, log, SnapshotManagerConfig{
		Namespace:   *namespace,
		PrefillSvc:  *prefillSvc,
		DecodeSvc:   *decodeSvc,
		SnapshotTTL: *snapshotTTL,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Watch Kubernetes Endpoints for both vLLM services.
	for _, svcName := range []string{*prefillSvc, *decodeSvc} {
		watcher := cache.NewListWatchFromClient(
			k8sClient.CoreV1().RESTClient(),
			"endpoints",
			*namespace,
			fields.OneTermEqualSelector("metadata.name", svcName),
		)
		informer := cache.NewSharedIndexInformer(watcher, &corev1.Endpoints{}, 30*time.Second, cache.Indexers{})
		informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { sm.Reconcile(ctx) },
			UpdateFunc: func(_, obj interface{}) { sm.Reconcile(ctx) },
			DeleteFunc: func(obj interface{}) { sm.Reconcile(ctx) },
		})
		go informer.Run(ctx.Done())
	}

	// Start gRPC xDS server.
	grpcServer := grpc.NewServer(
		grpc.MaxConcurrentStreams(1000),
		grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle: 5 * time.Minute,
			Time:              30 * time.Second,
			Timeout:           10 * time.Second,
		}),
	)

	callbacks := &xdsCallbacks{log: log}
	xdsSrv := server.NewServer(ctx, snapshotCache, callbacks)

	discoveryservice.RegisterAggregatedDiscoveryServiceServer(grpcServer, xdsSrv)
	endpointservice.RegisterEndpointDiscoveryServiceServer(grpcServer, xdsSrv)
	clusterservice.RegisterClusterDiscoveryServiceServer(grpcServer, xdsSrv)
	routeservice.RegisterRouteDiscoveryServiceServer(grpcServer, xdsSrv)
	listenerservice.RegisterListenerDiscoveryServiceServer(grpcServer, xdsSrv)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *xdsPort))
	if err != nil {
		log.Fatal("failed to listen", zap.Error(err))
	}

	log.Info("xDS control plane starting", zap.Int("port", *xdsPort))
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			log.Error("xDS server error", zap.Error(err))
		}
	}()

	// Initial snapshot push.
	sm.Reconcile(ctx)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Info("shutting down xDS server")
	grpcServer.GracefulStop()
}
