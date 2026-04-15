"""
vLLM KV-Cache Utilization Exporter

Scrapes the vLLM /metrics endpoint and re-exports KV-cache utilization
metrics in a format suitable for the metrics aggregation server and for
KEDA ScaledObjects (queue-depth-based autoscaling).

Exported metrics:
  vllm_kv_cache_usage_perc{node,model}    - GPU KV-cache blocks used (%)
  vllm_prefix_cache_hit_rate{node,model}  - Prefix cache hit rate (0-1)
  vllm_gpu_cache_hit_rate{node,model}     - GPU cache hit rate (0-1)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import httpx
from prometheus_client import Gauge, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kv-cache-exporter")

VLLM_METRICS_URL = os.environ.get(
    "VLLM_METRICS_URL", "http://localhost:8001/metrics"
)
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "9102"))
SCRAPE_INTERVAL = float(os.environ.get("SCRAPE_INTERVAL_SECONDS", "5"))
NODE_NAME = os.environ.get("NODE_NAME", os.environ.get("HOSTNAME", "unknown"))
MODEL_NAME = os.environ.get("MODEL_NAME", "unknown")

# ── Prometheus gauges ──────────────────────────────────────────────────────
KV_CACHE_USAGE = Gauge(
    "vllm_kv_cache_usage_perc",
    "GPU KV-cache blocks used (%)",
    ["node", "model"],
)
PREFIX_HIT_RATE = Gauge(
    "vllm_prefix_cache_hit_rate",
    "Prefix-cache (radix-tree) hit rate",
    ["node", "model"],
)
GPU_HIT_RATE = Gauge(
    "vllm_gpu_cache_hit_rate",
    "GPU KV-cache hit rate (vLLM native)",
    ["node", "model"],
)
SCRAPE_ERRORS = Gauge(
    "vllm_kv_exporter_scrape_errors_total",
    "Total scrape errors",
    ["node"],
)


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text format into {metric_name: value} dict."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{")[0].strip()
        try:
            result[name] = float(parts[1])
        except ValueError:
            pass
    return result


async def scrape_once(client: httpx.AsyncClient) -> Optional[dict[str, float]]:
    try:
        resp = await client.get(VLLM_METRICS_URL, timeout=3.0)
        resp.raise_for_status()
        return parse_prometheus_text(resp.text)
    except Exception as exc:
        logger.warning("Scrape error: %s", exc)
        SCRAPE_ERRORS.labels(node=NODE_NAME).inc()
        return None


async def main_loop() -> None:
    logger.info(
        "KV-cache exporter starting — scraping %s every %.1fs on port %d",
        VLLM_METRICS_URL,
        SCRAPE_INTERVAL,
        EXPORTER_PORT,
    )
    start_http_server(EXPORTER_PORT)

    async with httpx.AsyncClient() as client:
        while True:
            start = time.monotonic()
            metrics = await scrape_once(client)
            if metrics:
                kv_usage = metrics.get("vllm:gpu_cache_usage_perc", 0.0)
                prefix_hit = metrics.get("vllm:gpu_cache_hit_rate", 0.0)
                gpu_hit = metrics.get("vllm:gpu_cache_hit_rate", 0.0)

                KV_CACHE_USAGE.labels(node=NODE_NAME, model=MODEL_NAME).set(kv_usage)
                PREFIX_HIT_RATE.labels(node=NODE_NAME, model=MODEL_NAME).set(prefix_hit)
                GPU_HIT_RATE.labels(node=NODE_NAME, model=MODEL_NAME).set(gpu_hit)

                logger.debug(
                    "node=%s kv_usage=%.1f%% prefix_hit=%.3f",
                    NODE_NAME,
                    kv_usage,
                    prefix_hit,
                )

            elapsed = time.monotonic() - start
            await asyncio.sleep(max(0, SCRAPE_INTERVAL - elapsed))


if __name__ == "__main__":
    asyncio.run(main_loop())
