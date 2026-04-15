"""
OpenTelemetry Distributed Tracing

Instruments the inference gateway components with OpenTelemetry traces.
Each inference request gets a trace spanning:
  - Envoy ext_proc routing decision
  - Prefill phase (prompt → KV-cache)
  - KV-cache transfer (prefill → decode node)
  - Decode phase (token generation)

Trace context is propagated via W3C TraceContext headers (traceparent/tracestate).
"""

from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger("tracing")

OTLP_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://otel-collector.monitoring.svc.cluster.local:4317",
)
SERVICE = os.environ.get("OTEL_SERVICE_NAME", "inference-gateway")
PROPAGATOR = TraceContextTextMapPropagator()


def setup_tracing(service_name: str = SERVICE) -> TracerProvider:
    """
    Initialize the global OpenTelemetry TracerProvider with OTLP export.
    Call this once at application startup.
    """
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
    processor = BatchSpanProcessor(
        exporter,
        max_export_batch_size=512,
        export_timeout_millis=5000,
        schedule_delay_millis=500,
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    logger.info(
        "OpenTelemetry tracing configured: service=%s endpoint=%s",
        service_name,
        OTLP_ENDPOINT,
    )
    return provider


def get_tracer(component: str = SERVICE) -> trace.Tracer:
    """Return the OpenTelemetry tracer for the given component."""
    return trace.get_tracer(component, schema_url="https://opentelemetry.io/schemas/1.24.0")


@contextmanager
def inference_span(
    tracer: trace.Tracer,
    name: str,
    request_id: str,
    parent_context: Optional[Any] = None,
    **attributes: Any,
) -> Generator[trace.Span, None, None]:
    """
    Context manager that creates a span for an inference gateway operation.

    Usage:
        with inference_span(tracer, "prefill", request_id="abc") as span:
            span.set_attribute("prompt_tokens", 512)
            # ... do work ...
    """
    ctx = parent_context
    with tracer.start_as_current_span(
        name,
        context=ctx,
        kind=SpanKind.INTERNAL,
        attributes={
            "request.id": request_id,
            "inference.component": name,
            **{str(k): str(v) for k, v in attributes.items()},
        },
    ) as span:
        start = time.perf_counter()
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("duration_ms", elapsed_ms)


def extract_trace_context(headers: dict[str, str]) -> Any:
    """Extract W3C TraceContext from HTTP headers."""
    return PROPAGATOR.extract(headers)


def inject_trace_context(span_context: Any, headers: dict[str, str]) -> dict[str, str]:
    """Inject W3C TraceContext into HTTP headers for downstream propagation."""
    PROPAGATOR.inject(headers, context=span_context)
    return headers


def trace_async(name: str):
    """
    Decorator that wraps an async function with an OpenTelemetry span.

    Usage:
        @trace_async("prefill_phase")
        async def run_prefill(request_id: str, prompt: str) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                name,
                kind=SpanKind.INTERNAL,
                attributes={"function": func.__qualname__},
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise
        return wrapper
    return decorator


# ── Span attribute constants ───────────────────────────────────────────────

class InferenceAttributes:
    """Standard attribute keys for inference gateway spans."""

    REQUEST_ID = "inference.request_id"
    TENANT_ID = "inference.tenant_id"
    MODEL_NAME = "inference.model_name"
    PROMPT_TOKENS = "inference.prompt_tokens"
    COMPLETION_TOKENS = "inference.completion_tokens"
    PREFIX_CACHE_HIT = "inference.prefix_cache_hit"
    ROUTING_CLUSTER = "inference.routing_cluster"
    ROUTING_NODE = "inference.routing_node"
    ROUTING_SCORE = "inference.routing_score"
    KV_TRANSFER_BYTES = "inference.kv_transfer_bytes"
    KV_TRANSFER_NODE = "inference.kv_transfer_target_node"
    TTFT_MS = "inference.ttft_ms"
    PREFILL_LATENCY_MS = "inference.prefill_latency_ms"
    DECODE_LATENCY_MS = "inference.decode_latency_ms"
