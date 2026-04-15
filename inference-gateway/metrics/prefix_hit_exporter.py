"""
vLLM Prefix Cache Hit Rate Exporter

Exports per-node prefix-cache hit rate metrics. This exporter reads from:
  1. The vLLM /metrics endpoint (gpu_cache_hit_rate — radix-tree hit rate).
  2. The distributed Redis prefix cache (query counts per node).

The hit rate is used by the routing engine to prefer nodes with a warm
prefix cache for the incoming request's system prompt / conversation context.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Deque

import httpx
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prefix-hit-exporter")

VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://localhost:8001/metrics")
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "9104"))
SCRAPE_INTERVAL = float(os.environ.get("SCRAPE_INTERVAL_SECONDS", "5"))
NODE_NAME = os.environ.get("NODE_NAME", os.environ.get("HOSTNAME", "unknown"))
MODEL_NAME = os.environ.get("MODEL_NAME", "unknown")
# Rolling window for hit rate computation (in seconds).
ROLLING_WINDOW_SECONDS = int(os.environ.get("ROLLING_WINDOW_SECONDS", "60"))

# ── Prometheus metrics ─────────────────────────────────────────────────────
PREFIX_CACHE_HIT_RATE = Gauge(
    "vllm_prefix_cache_hit_rate",
    "Rolling prefix-cache hit rate over the last N seconds",
    ["node", "model"],
)
PREFIX_CACHE_HITS_TOTAL = Counter(
    "vllm_prefix_cache_hits_total",
    "Total number of prefix-cache hits",
    ["node"],
)
PREFIX_CACHE_MISSES_TOTAL = Counter(
    "vllm_prefix_cache_misses_total",
    "Total number of prefix-cache misses",
    ["node"],
)
PREFIX_CACHE_BLOCK_COUNT = Gauge(
    "vllm_prefix_cache_block_count",
    "Number of prefix-cached KV-cache blocks currently resident on this node",
    ["node"],
)
PREFIX_HIT_LATENCY_SAVED = Histogram(
    "vllm_prefix_cache_prefill_time_saved_seconds",
    "Estimated prefill time saved by prefix-cache hits",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# ── Rolling hit rate ────────────────────────────────────────────────────────


class RollingHitRate:
    """Tracks prefix cache hits/misses over a rolling time window."""

    def __init__(self, window_seconds: int) -> None:
        self._window = window_seconds
        # Each entry: (timestamp, is_hit)
        self._events: Deque[tuple[float, bool]] = deque()

    def record(self, is_hit: bool) -> None:
        now = time.monotonic()
        self._events.append((now, is_hit))
        self._evict_old(now)

    def hit_rate(self) -> float:
        now = time.monotonic()
        self._evict_old(now)
        if not self._events:
            return 0.0
        hits = sum(1 for _, is_hit in self._events if is_hit)
        return hits / len(self._events)

    def _evict_old(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


_rolling = RollingHitRate(ROLLING_WINDOW_SECONDS)
_prev_gpu_hit_rate: float = 0.0
_prev_scrape_total: float = 0.0


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
    global _prev_gpu_hit_rate, _prev_scrape_total

    logger.info(
        "Prefix hit exporter starting — scraping %s every %.1fs on port %d",
        VLLM_METRICS_URL,
        SCRAPE_INTERVAL,
        EXPORTER_PORT,
    )
    start_http_server(EXPORTER_PORT)

    async with httpx.AsyncClient() as client:
        while True:
            loop_start = time.monotonic()
            try:
                resp = await client.get(VLLM_METRICS_URL, timeout=3.0)
                resp.raise_for_status()
                metrics = parse_prometheus_text(resp.text)

                gpu_hit_rate = metrics.get("vllm:gpu_cache_hit_rate", 0.0)
                num_requests = metrics.get("vllm:request_success_total", 0.0)

                # Derive hits/misses from delta in request count * hit rate.
                delta_requests = max(0, num_requests - _prev_scrape_total)
                if delta_requests > 0:
                    delta_hits = delta_requests * gpu_hit_rate
                    delta_misses = delta_requests * (1 - gpu_hit_rate)
                    for _ in range(int(delta_hits)):
                        _rolling.record(True)
                        PREFIX_CACHE_HITS_TOTAL.labels(node=NODE_NAME).inc()
                    for _ in range(int(delta_misses)):
                        _rolling.record(False)
                        PREFIX_CACHE_MISSES_TOTAL.labels(node=NODE_NAME).inc()

                _prev_scrape_total = num_requests
                _prev_gpu_hit_rate = gpu_hit_rate

                rolling_rate = _rolling.hit_rate()
                PREFIX_CACHE_HIT_RATE.labels(node=NODE_NAME, model=MODEL_NAME).set(rolling_rate)

                logger.debug(
                    "node=%s gpu_hit_rate=%.3f rolling_hit_rate=%.3f",
                    NODE_NAME,
                    gpu_hit_rate,
                    rolling_rate,
                )

            except Exception as exc:
                logger.warning("Scrape error: %s", exc)

            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0, SCRAPE_INTERVAL - elapsed))


if __name__ == "__main__":
    asyncio.run(main_loop())
