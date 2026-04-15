# Envoy-Based LLM Inference Gateway

Production-grade inference gateway routing LLM traffic across vLLM fleets using real-time
KV-cache utilization, queue depth, and prefix-cache hit metrics — designed for 1M+ customers.

## Architecture

```
Clients (1M+)
     │
     ▼
[Global Load Balancer / Anycast]
     │
     ▼
[Envoy Gateway Tier]  ◄──── xDS Control Plane (Go/gRPC)
     │  (ext_proc + dynamic routing)
     ▼
[Routing Decision Engine]  ◄──── Real-time Metrics (KV-cache %, queue depth, prefix hits)
     │
     ├──► vLLM Prefill Fleet (KubeRay Workers, GPU H100 SXM5)
     │
     └──► vLLM Decode Fleet  (KubeRay Workers, GPU A100 PCIe)
                │
                ▼
         [KV-Cache Layer / Prefix Cache Store (Redis Cluster + RDMA)]
```

## Components

| Directory | Description |
|-----------|-------------|
| `envoy/` | Envoy bootstrap config, xDS control plane, ext_proc filter |
| `routing/` | Routing engine and strategies (KV-cache aware, prefix-cache, queue-depth) |
| `vllm/` | Prefill/decode disaggregated vLLM worker configs |
| `prefix_cache/` | GPU KV-cache block manager, radix-tree, eviction policies |
| `metrics/` | Prometheus exporters for vLLM metrics |
| `kuberay/` | KubeRay RayCluster manifests and autoscaler config |
| `sampling/` | Sampling strategies, speculative decoding, continuous batching |
| `observability/` | Grafana dashboards, Prometheus alerts, OpenTelemetry tracing |
| `gateway-api/` | Kubernetes Gateway API manifests |
| `deploy/` | Helm chart, Terraform, Kustomize overlays |

## Quick Start

### Prerequisites
- Kubernetes 1.28+ cluster with GPU node pools (H100/A100)
- KubeRay operator installed
- Envoy Gateway installed
- Prometheus + Grafana stack

### Deploy

```bash
# 1. Deploy KubeRay cluster
kubectl apply -f kuberay/ray_cluster.yaml

# 2. Deploy Envoy xDS control plane
cd envoy/xds && go build -o xds-server . && kubectl apply -f ../../deploy/kustomize/base/

# 3. Apply Gateway API manifests
kubectl apply -f gateway-api/

# 4. Deploy metrics exporters
kubectl apply -f metrics/

# 5. Install via Helm (full stack)
helm install inference-gateway deploy/helm/ -f deploy/helm/values.yaml
```

## SLO Targets

| Metric | Target |
|--------|--------|
| Time-to-First-Token (p99) | < 500ms |
| Inter-Token Latency (p99) | < 50ms |
| Error rate | < 0.1% |
| Availability | 99.99% |
| Prefix cache hit rate | > 60% |

## Routing Strategy

The composite routing score per node is computed as:

```
score(node) = α·prefix_hit_rate(node)
            + β·(1 - kv_cache_util(node))
            + γ·(1 - normalized_queue_depth(node))
```

Default weights: α=0.4, β=0.35, γ=0.25

## Prefill/Decode Disaggregation

LLM inference separates into two phases:

- **Prefill**: Compute-bound. Processes all input tokens in a single forward pass.
  Runs on H100 SXM5 GPUs (high FLOP/s). Produces the KV-cache.
- **Decode**: Memory-bandwidth-bound. Autoregressively generates one token per step.
  Runs on A100 PCIe GPUs. Reads the full KV-cache each step.

KV-cache tensors are transferred from prefill → decode workers via RDMA over InfiniBand
(~200 GB/s) or NVLink P2P, serialized to FP8 for compression.
