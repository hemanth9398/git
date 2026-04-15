"""
Prefill/Decode Disaggregation Orchestrator

This module implements the disaggregated serving controller that:
  1. Receives an inference request from the gateway.
  2. Routes it to the best prefill worker (based on KV-cache hit).
  3. Waits for the prefill worker to complete the forward pass and
     transfer the KV-cache blocks to the selected decode worker.
  4. Forwards the decode request (with pre-filled KV pointer) to the
     decode worker.
  5. Streams token output back to the caller.

The KV-cache transfer is performed directly between the prefill and
decode worker GPUs via NCCL/RDMA — this module just orchestrates the
handshake.

Architecture:
  Client → Gateway (Envoy) → ext_proc → Orchestrator
                                              │
                               ┌──────────────┴──────────────┐
                               ▼                             ▼
                       Prefill Worker               Decode Worker
                    (KV-cache producer)         (KV-cache consumer)
                               │                             │
                               └────── RDMA KV transfer ────┘
                                        (NCCL/UCX/TCP)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("disaggregated-orchestrator")

PREFILL_SERVICE = os.environ.get(
    "PREFILL_SERVICE", "http://vllm-prefill.inference-gateway.svc.cluster.local:8000"
)
DECODE_SERVICE = os.environ.get(
    "DECODE_SERVICE", "http://vllm-decode.inference-gateway.svc.cluster.local:8000"
)
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "300"))


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    stream: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""


@dataclass
class PrefillResult:
    request_id: str
    prompt_token_ids: list[int]
    kv_transfer_token_ids: list[int]
    prefill_node: str
    decode_node: str
    prefill_latency_ms: float


class DisaggregatedOrchestrator:
    """Coordinates prefill-decode disaggregated LLM inference."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200),
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def infer(self, req: InferenceRequest) -> dict:
        """
        Run disaggregated inference:
          1. Send to prefill worker.
          2. Prefill worker transfers KV-cache to decode worker.
          3. Receive completion from decode worker.
        """
        start = time.perf_counter()

        # Step 1: Prefill phase — compute KV-cache for input tokens.
        prefill_payload = {
            "prompt": req.prompt,
            "max_tokens": 1,  # Prefill generates 0 or 1 token; decode does the rest
            "temperature": req.temperature,
            "stream": False,
            "request_id": req.request_id,
            # Signal to the prefill worker to transfer KV-cache to the decode worker.
            "kv_transfer_target": DECODE_SERVICE,
        }
        logger.info(
            "Starting prefill for request %s (len=%d chars)",
            req.request_id,
            len(req.prompt),
        )

        prefill_resp = await self._http.post(
            f"{PREFILL_SERVICE}/v1/completions", json=prefill_payload
        )
        prefill_resp.raise_for_status()
        prefill_data = prefill_resp.json()
        prefill_latency = (time.perf_counter() - start) * 1000

        logger.info(
            "Prefill complete for %s in %.1fms", req.request_id, prefill_latency
        )

        # Step 2: Decode phase — generate tokens using pre-filled KV-cache.
        decode_payload = {
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "min_p": req.min_p,
            "stream": req.stream,
            "request_id": req.request_id,
            # KV-cache was transferred in step 1; decode worker should find it.
            "kv_transfer_token_ids": prefill_data.get("usage", {}).get(
                "prompt_tokens", 0
            ),
        }

        decode_resp = await self._http.post(
            f"{DECODE_SERVICE}/v1/completions", json=decode_payload
        )
        decode_resp.raise_for_status()
        result = decode_resp.json()

        total_latency = (time.perf_counter() - start) * 1000
        logger.info("Request %s completed in %.1fms", req.request_id, total_latency)
        return result

    async def infer_stream(self, req: InferenceRequest) -> AsyncIterator[str]:
        """
        Streaming variant: prefill then stream decode tokens.
        """
        # Step 1: Prefill (non-streaming).
        prefill_payload = {
            "prompt": req.prompt,
            "max_tokens": 1,
            "stream": False,
            "request_id": req.request_id,
            "kv_transfer_target": DECODE_SERVICE,
        }
        async with self._http.stream(
            "POST", f"{PREFILL_SERVICE}/v1/completions", json=prefill_payload
        ) as resp:
            resp.raise_for_status()
            await resp.aread()

        # Step 2: Decode (streaming).
        decode_payload = {
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "min_p": req.min_p,
            "stream": True,
            "request_id": req.request_id,
        }
        async with self._http.stream(
            "POST", f"{DECODE_SERVICE}/v1/completions", json=decode_payload
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_text():
                yield chunk


# Module-level singleton for use by the FastAPI app.
_orchestrator: DisaggregatedOrchestrator | None = None


def get_orchestrator() -> DisaggregatedOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DisaggregatedOrchestrator()
    return _orchestrator
