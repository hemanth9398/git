"""
vLLM Queue Depth Exporter

Exports per-node request queue depth as a Prometheus gauge. This metric
is used by:
  1. The routing engine to avoid sending requests to overloaded nodes.
  2. KEDA ScaledObjects for queue-depth-based autoscaling.
  3. Envoy rate-limit responses (shed load when queue > threshold).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx
from prometheus_client import Gauge, Counter, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("queue-exporter")

VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://localhost:8001/metrics")
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "9103"))
SCRAPE_INTERVAL = float(os.environ.get("SCRAPE_INTERVAL_SECONDS", "2"))
NODE_NAME = os.environ.get("NODE_NAME", os.environ.get("HOSTNAME", "unknown"))
MODEL_NAME = os.environ.get("MODEL_NAME", "unknown")
# Alert threshold: log a warning when queue depth exceeds this value.
QUEUE_ALERT_THRESHOLD = int(os.environ.get("QUEUE_ALERT_THRESHOLD", "50"))

# ── Prometheus metrics ─────────────────────────────────────────────────────
REQUESTS_WAITING = Gauge(
    "vllm_num_requests_waiting",
    "Number of requests currently waiting in the queue",
    ["node", "model"],
)
REQUESTS_RUNNING = Gauge(
    "vllm_num_requests_running",
    "Number of requests currently being processed",
    ["node", "model"],
)
REQUESTS_SWAPPED = Gauge(
    "vllm_num_requests_swapped",
    "Number of requests with KV-cache swapped to CPU",
    ["node", "model"],
)
QUEUE_SATURATION = Gauge(
    "vllm_queue_saturation_ratio",
    "Queue depth as fraction of saturation threshold (0-1+)",
    ["node"],
)
SCRAPE_ERRORS = Counter(
    "vllm_queue_exporter_scrape_errors_total",
    "Total scrape errors",
    ["node"],
)


def parse_prometheus_text(text: str) -> dict[str, float]:
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


async def main_loop() -> None:
    logger.info(
        "Queue exporter starting — scraping %s every %.1fs on port %d",
        VLLM_METRICS_URL,
        SCRAPE_INTERVAL,
        EXPORTER_PORT,
    )
    start_http_server(EXPORTER_PORT)

    async with httpx.AsyncClient() as client:
        while True:
            start = time.monotonic()
            try:
                resp = await client.get(VLLM_METRICS_URL, timeout=2.0)
                resp.raise_for_status()
                metrics = parse_prometheus_text(resp.text)

                waiting = metrics.get("vllm:num_requests_waiting", 0.0)
                running = metrics.get("vllm:num_requests_running", 0.0)
                swapped = metrics.get("vllm:num_requests_swapped", 0.0)

                REQUESTS_WAITING.labels(node=NODE_NAME, model=MODEL_NAME).set(waiting)
                REQUESTS_RUNNING.labels(node=NODE_NAME, model=MODEL_NAME).set(running)
                REQUESTS_SWAPPED.labels(node=NODE_NAME, model=MODEL_NAME).set(swapped)
                QUEUE_SATURATION.labels(node=NODE_NAME).set(
                    waiting / QUEUE_ALERT_THRESHOLD
                )

                if waiting > QUEUE_ALERT_THRESHOLD:
                    logger.warning(
                        "Queue depth alert: node=%s waiting=%d (threshold=%d)",
                        NODE_NAME,
                        int(waiting),
                        QUEUE_ALERT_THRESHOLD,
                    )

            except Exception as exc:
                logger.warning("Scrape error: %s", exc)
                SCRAPE_ERRORS.labels(node=NODE_NAME).inc()

            elapsed = time.monotonic() - start
            await asyncio.sleep(max(0, SCRAPE_INTERVAL - elapsed))


if __name__ == "__main__":
    asyncio.run(main_loop())
