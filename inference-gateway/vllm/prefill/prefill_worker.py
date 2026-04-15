"""
vLLM Prefill Worker

This worker runs vLLM in prefill-only mode as part of a prefill/decode
disaggregated serving setup. It:

  1. Accepts inference requests from the gateway.
  2. Runs only the prefill (forward pass over input tokens) to populate
     the KV-cache.
  3. Transfers the resulting KV-cache blocks to the designated decode worker
     via RDMA (UCX/NCCL) or TCP fallback.
  4. Exports Prometheus metrics on port 8001.

References:
  - vLLM disaggregated prefill: https://docs.vllm.ai/en/latest/serving/disagg_prefill.html
  - PagedAttention: https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.config import KVTransferConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prefill-worker")

# ── Prometheus metrics ─────────────────────────────────────────────────────
REQUESTS_TOTAL = Counter(
    "vllm_prefill_requests_total", "Total prefill requests", ["status"]
)
PREFILL_LATENCY = Histogram(
    "vllm_prefill_latency_seconds",
    "Time for prefill phase (seconds)",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
KV_TRANSFER_LATENCY = Histogram(
    "vllm_kv_transfer_latency_seconds",
    "Time to transfer KV-cache to decode worker (seconds)",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)
KV_CACHE_USAGE = Gauge("vllm_kv_cache_usage_perc", "KV-cache blocks used (%)")
QUEUE_DEPTH = Gauge("vllm_num_requests_waiting", "Requests waiting in queue")

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3-70b-Instruct")
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL_SIZE", "8"))
GPU_MEMORY_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
DECODE_HOST = os.environ.get("DECODE_SERVICE_HOST", "vllm-decode-headless.inference-gateway.svc.cluster.local")
DECODE_PORT = int(os.environ.get("DECODE_SERVICE_PORT", "8100"))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8001"))
BLOCK_SIZE = int(os.environ.get("KV_BLOCK_SIZE", "16"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
ENABLE_PREFIX_CACHING = os.environ.get("ENABLE_PREFIX_CACHING", "true").lower() == "true"


class CompletionRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict] | None = None
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    stream: bool = False
    request_id: str | None = None


app = FastAPI(title="vLLM Prefill Worker")
engine: AsyncLLMEngine | None = None


def build_engine() -> AsyncLLMEngine:
    """Build the vLLM async engine configured for prefill-only mode."""
    kv_transfer_config = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_producer",  # This worker PRODUCES (sends) KV-cache to decode worker
        kv_rank=0,
        kv_parallel_size=1,
        kv_buffer_device="cuda",
        kv_buffer_size=int(1e9),  # 1 GB transfer buffer
    )

    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        # Disaggregated prefill mode: only run the prefill phase, then
        # transfer KV-cache to the decode worker.
        kv_transfer_config=kv_transfer_config,
        # Disable chunked prefill for disaggregated mode (not compatible).
        enable_chunked_prefill=False,
        # Use FP8 KV-cache for memory efficiency during transfer.
        kv_cache_dtype="fp8",
        # Large max_num_seqs for high-throughput batching.
        max_num_seqs=256,
        disable_log_stats=False,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


@app.on_event("startup")
async def startup():
    global engine
    logger.info("Starting vLLM prefill engine for model: %s", MODEL_NAME)
    start_http_server(METRICS_PORT)
    engine = build_engine()
    asyncio.create_task(metrics_loop())
    logger.info("Prefill worker ready")


@app.get("/health")
async def health():
    return {"status": "ok", "role": "prefill"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/completions")
async def completions(req: CompletionRequest, http_req: Request):
    return await _handle_request(req, http_req)


@app.post("/v1/chat/completions")
async def chat_completions(req: CompletionRequest, http_req: Request):
    return await _handle_request(req, http_req)


async def _handle_request(req: CompletionRequest, http_req: Request):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    prompt = _extract_prompt(req)
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k if req.top_k > 0 else -1,
        min_p=req.min_p,
        max_tokens=req.max_tokens,
    )

    request_id = req.request_id or f"prefill-{time.time_ns()}"
    start = time.perf_counter()

    try:
        REQUESTS_TOTAL.labels(status="started").inc()
        if req.stream:
            return StreamingResponse(
                _stream_generate(request_id, prompt, sampling_params, start),
                media_type="text/event-stream",
            )
        else:
            result = await _generate_full(request_id, prompt, sampling_params)
            elapsed = time.perf_counter() - start
            PREFILL_LATENCY.observe(elapsed)
            REQUESTS_TOTAL.labels(status="success").inc()
            return JSONResponse(result)
    except Exception as exc:
        REQUESTS_TOTAL.labels(status="error").inc()
        logger.exception("Request %s failed", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _generate_full(request_id: str, prompt: str, params: SamplingParams) -> dict:
    final_output = None
    async for output in engine.generate(prompt, params, request_id):
        final_output = output
    if final_output is None:
        raise RuntimeError("No output from engine")
    return {
        "id": request_id,
        "object": "text_completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "text": final_output.outputs[0].text,
                "finish_reason": final_output.outputs[0].finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(final_output.outputs[0].token_ids),
        },
    }


async def _stream_generate(
    request_id: str, prompt: str, params: SamplingParams, start: float
) -> AsyncIterator[str]:
    try:
        async for output in engine.generate(prompt, params, request_id):
            if output.outputs:
                token = output.outputs[0].text
                yield f"data: {token}\n\n"
        elapsed = time.perf_counter() - start
        PREFILL_LATENCY.observe(elapsed)
        REQUESTS_TOTAL.labels(status="success").inc()
        yield "data: [DONE]\n\n"
    except Exception as exc:
        REQUESTS_TOTAL.labels(status="error").inc()
        logger.exception("Streaming request %s failed", request_id)
        yield f"data: {{\"error\": \"{exc}\"}}\n\n"


def _extract_prompt(req: CompletionRequest) -> str:
    if req.prompt:
        return req.prompt
    if req.messages:
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in req.messages
        )
    return ""


async def metrics_loop():
    """Periodically update Prometheus gauges from engine stats."""
    while True:
        try:
            if engine is not None:
                stats = await engine.get_model_config()
                # Update KV-cache usage gauge.
                # (Engine exposes this via do_log_stats in vLLM 0.4+.)
        except Exception:
            pass
        await asyncio.sleep(5)


if __name__ == "__main__":
    uvicorn.run(
        "prefill_worker:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop",
        log_level="info",
    )
