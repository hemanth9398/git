"""
Continuous Batching Scheduler

Implements iteration-level scheduling (continuous batching / "in-flight batching"):
requests are dynamically added to or removed from the active batch at every
decode iteration, rather than waiting for an entire batch to complete.

This achieves significantly higher GPU utilization compared to static batching
because:
  - Short requests complete and free slots immediately.
  - New arrivals can join the batch as soon as a slot opens.
  - The batch size adapts to available KV-cache memory every iteration.

References:
  - Orca: "A Distributed Serving System for Transformer-Based Generative Models"
    (Yu et al., OSDI 2022) https://www.usenix.org/conference/osdi22/presentation/yu
  - vLLM: "Efficient Memory Management for LLM Serving with PagedAttention"
    (Kwon et al., SOSP 2023) https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Callable, Optional

logger = logging.getLogger("continuous-batching")


class SequenceStatus(Enum):
    WAITING = auto()    # In queue, not yet scheduled
    RUNNING = auto()    # In active batch, being decoded
    FINISHED = auto()   # Generation complete (EOS or max_tokens)
    ABORTED = auto()    # Cancelled by client


@dataclass
class Sequence:
    """A single inference sequence managed by the scheduler."""

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    generated_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    arrival_time: float = field(default_factory=time.monotonic)
    # Number of KV-cache blocks allocated for this sequence.
    num_blocks: int = 0
    # Callback invoked each time a new token is generated.
    on_token: Optional[Callable[[int], None]] = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED)


@dataclass
class SchedulerConfig:
    """Configuration for the continuous batching scheduler."""

    # Maximum number of sequences in the active batch.
    max_batch_size: int = 256
    # Maximum number of new tokens generated per iteration (across all sequences).
    max_tokens_per_iter: int = 4096
    # Maximum KV-cache blocks available (controls memory-aware scheduling).
    max_kv_blocks: int = 10000
    # Number of KV-cache blocks per token.
    block_size: int = 16
    # Delay between decode iterations (0 = run as fast as possible).
    iteration_delay_ms: float = 0.0
    # Maximum waiting time before a request is forcibly scheduled (anti-starvation).
    max_wait_seconds: float = 30.0


class ContinuousBatchingScheduler:
    """
    Iteration-level scheduler for continuous batching.

    The scheduler maintains three queues:
      - waiting:  requests not yet admitted to the batch
      - running:  requests in the current active batch
      - finished: completed requests (held briefly for result retrieval)

    On each iteration:
      1. Preempt running sequences if KV-cache is too full (swap to CPU or abort).
      2. Promote waiting sequences into the batch (up to max_batch_size).
      3. Execute one decode step (yield to the model execution layer).
      4. Remove finished sequences.
      5. Repeat.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self._waiting: list[Sequence] = []
        self._running: list[Sequence] = []
        self._finished: dict[str, Sequence] = {}
        self._lock = asyncio.Lock()
        self._step_count: int = 0
        self._total_tokens_generated: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    async def add_request(self, seq: Sequence) -> None:
        """Add a new request to the waiting queue."""
        async with self._lock:
            self._waiting.append(seq)
            logger.debug(
                "Queued request %s (waiting=%d, running=%d)",
                seq.request_id, len(self._waiting), len(self._running),
            )

    async def abort_request(self, request_id: str) -> None:
        """Cancel a request that is waiting or running."""
        async with self._lock:
            for seq in self._waiting:
                if seq.request_id == request_id:
                    seq.status = SequenceStatus.ABORTED
                    self._waiting.remove(seq)
                    return
            for seq in self._running:
                if seq.request_id == request_id:
                    seq.status = SequenceStatus.ABORTED
                    return

    async def get_result(self, request_id: str, timeout: float = 30.0) -> Sequence:
        """Wait for a request to finish and return the completed Sequence."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            async with self._lock:
                seq = self._finished.pop(request_id, None)
            if seq is not None:
                return seq
            await asyncio.sleep(0.01)
        raise TimeoutError(f"Request {request_id} did not complete within {timeout}s")

    # ── Scheduling logic ──────────────────────────────────────────────────

    async def schedule_iteration(
        self,
        model_step_fn: Callable[[list[Sequence]], dict[str, int]],
    ) -> None:
        """
        Execute one scheduler iteration:
          1. Admit waiting sequences into the running batch.
          2. Call model_step_fn with the running batch.
          3. Process outputs (append tokens, detect EOS).
          4. Move finished sequences out of the running batch.

        Args:
            model_step_fn: Callable that takes the running batch and returns
                           a dict mapping request_id → next_token_id.
        """
        async with self._lock:
            self._admit_sequences()
            batch = [s for s in self._running if s.status == SequenceStatus.RUNNING]

        if not batch:
            return

        # Call into the model layer (outside the lock to allow concurrency).
        token_map = model_step_fn(batch)

        async with self._lock:
            finished = []
            for seq in batch:
                new_token = token_map.get(seq.request_id)
                if new_token is None:
                    continue
                seq.generated_token_ids.append(new_token)
                self._total_tokens_generated += 1
                if seq.on_token:
                    seq.on_token(new_token)
                # Check termination conditions.
                if (
                    new_token == 2  # EOS (hardcoded; replace with tokenizer EOS)
                    or len(seq.generated_token_ids) >= seq.max_new_tokens
                ):
                    seq.status = SequenceStatus.FINISHED
                    finished.append(seq)

            for seq in finished:
                self._running.remove(seq)
                self._finished[seq.request_id] = seq

            self._step_count += 1

        if self._config.iteration_delay_ms > 0:
            await asyncio.sleep(self._config.iteration_delay_ms / 1000.0)

    def _admit_sequences(self) -> None:
        """
        Move waiting sequences into the running batch while:
          - batch size ≤ max_batch_size
          - estimated KV-cache usage ≤ max_kv_blocks
        """
        available_slots = self._config.max_batch_size - len(self._running)
        used_blocks = sum(s.num_blocks for s in self._running)

        new_arrivals: list[Sequence] = []
        remaining: list[Sequence] = []

        # Sort waiting queue by arrival time (FCFS) with starvation protection.
        now = time.monotonic()
        self._waiting.sort(key=lambda s: s.arrival_time)

        for seq in self._waiting:
            if available_slots <= 0:
                remaining.append(seq)
                continue
            # Estimate KV-cache blocks needed for this sequence.
            needed_blocks = (seq.num_tokens + self._config.block_size - 1) // self._config.block_size
            if used_blocks + needed_blocks > self._config.max_kv_blocks:
                # Check if this request has been waiting too long (anti-starvation).
                if now - seq.arrival_time < self._config.max_wait_seconds:
                    remaining.append(seq)
                    continue
                # Force-admit even if memory is tight (anti-starvation).
                logger.warning(
                    "Force-admitting stale request %s (waited %.1fs)",
                    seq.request_id, now - seq.arrival_time,
                )

            seq.status = SequenceStatus.RUNNING
            seq.num_blocks = needed_blocks
            new_arrivals.append(seq)
            used_blocks += needed_blocks
            available_slots -= 1

        self._running.extend(new_arrivals)
        self._waiting = remaining

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "step_count": self._step_count,
            "waiting": len(self._waiting),
            "running": len(self._running),
            "finished": len(self._finished),
            "total_tokens_generated": self._total_tokens_generated,
        }
