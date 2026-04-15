"""
GPU KV-Cache Block Manager

Manages the allocation and tracking of KV-cache blocks on GPU memory using
PagedAttention's block abstraction. Each block stores the key/value tensors
for a fixed number of tokens (block_size).

Key concepts:
  - Physical blocks: fixed-size GPU memory regions holding KV tensors.
  - Logical blocks: request-level view mapping token positions → physical blocks.
  - Reference counting: blocks shared across prefix-cached requests are
    ref-counted; only freed when count drops to 0.
  - Eviction: LRU eviction of unreferenced blocks when GPU memory is under pressure.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import torch

# Default block size matches vLLM default (16 tokens per block).
DEFAULT_BLOCK_SIZE = 16


@dataclass
class KVBlock:
    """A single KV-cache block occupying contiguous GPU memory."""

    block_id: int
    device: torch.device
    # FP8 key/value tensors: [2, num_heads, block_size, head_dim]
    data: Optional[torch.Tensor] = None
    ref_count: int = 0
    last_access: float = field(default_factory=time.monotonic)
    # Hash of the token IDs stored in this block (for prefix-cache lookup).
    token_hash: int = 0
    # Whether this block is currently pinned (active request is using it).
    pinned: bool = False


class BlockAllocator:
    """
    Thread-safe allocator for GPU KV-cache blocks.

    Maintains a free list and an LRU eviction pool of unreferenced blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        self._lock = threading.Lock()
        # Pre-allocate the entire KV-cache as a single tensor:
        # Shape: [num_blocks, 2 (K+V), num_heads, block_size, head_dim]
        self._pool: torch.Tensor = torch.zeros(
            (num_blocks, 2, num_heads, block_size, head_dim),
            dtype=dtype,
            device=self.device,
        )

        # Free list of block IDs.
        self._free: list[int] = list(range(num_blocks))
        # LRU cache: block_id → KVBlock (ordered by least-recently-used first).
        self._lru: OrderedDict[int, KVBlock] = OrderedDict()
        # Active blocks: block_id → KVBlock.
        self._active: dict[int, KVBlock] = {}

    # ── Allocation ──────────────────────────────────────────────────────────

    def allocate(self, token_hash: int = 0) -> KVBlock:
        """
        Allocate a free block. If no free blocks are available, evict the
        least-recently-used unreferenced block from the LRU cache.

        Raises RuntimeError if all blocks are pinned (OOM).
        """
        with self._lock:
            block_id = self._get_free_block()
            block = KVBlock(
                block_id=block_id,
                device=self.device,
                data=self._pool[block_id],
                ref_count=1,
                last_access=time.monotonic(),
                token_hash=token_hash,
                pinned=True,
            )
            self._active[block_id] = block
            return block

    def _get_free_block(self) -> int:
        if self._free:
            return self._free.pop()
        # Try to evict from LRU (unreferenced, unpinned blocks).
        for block_id, block in self._lru.items():
            if not block.pinned and block.ref_count == 0:
                del self._lru[block_id]
                del self._active[block_id]
                return block_id
        raise RuntimeError(
            f"GPU KV-cache OOM: all {self.num_blocks} blocks are pinned"
        )

    # ── Reference counting ──────────────────────────────────────────────────

    def incref(self, block: KVBlock) -> None:
        """Increment the reference count of a block (shared prefix reuse)."""
        with self._lock:
            block.ref_count += 1
            block.last_access = time.monotonic()
            # Move to end of LRU (most recently used).
            if block.block_id in self._lru:
                self._lru.move_to_end(block.block_id)

    def decref(self, block: KVBlock) -> None:
        """Decrement the reference count. Moves to LRU pool when count hits 0."""
        with self._lock:
            block.ref_count = max(0, block.ref_count - 1)
            if block.ref_count == 0:
                block.pinned = False
                block.last_access = time.monotonic()
                self._lru[block.block_id] = block

    # ── Stats ───────────────────────────────────────────────────────────────

    def usage_percent(self) -> float:
        """Return percentage of blocks currently in use (allocated or LRU)."""
        with self._lock:
            used = self.num_blocks - len(self._free)
            return 100.0 * used / self.num_blocks

    def free_blocks(self) -> int:
        with self._lock:
            return len(self._free)

    def evictable_blocks(self) -> int:
        """Number of unreferenced (evictable) blocks in the LRU pool."""
        with self._lock:
            return sum(1 for b in self._lru.values() if not b.pinned and b.ref_count == 0)


class GPUCacheManager:
    """
    High-level KV-cache manager that combines block allocation with prefix-
    cache awareness. It maintains a mapping from token-sequence hashes to
    allocated block IDs, enabling prefix reuse across requests.
    """

    def __init__(self, allocator: BlockAllocator) -> None:
        self._allocator = allocator
        self._lock = threading.Lock()
        # token_hash → block_id mapping for prefix cache.
        self._prefix_map: dict[int, int] = {}

    def get_or_allocate(self, token_hash: int) -> tuple[KVBlock, bool]:
        """
        Return the block for token_hash if cached (hit), otherwise allocate
        a new block (miss). The bool indicates whether this was a cache hit.
        """
        with self._lock:
            if token_hash in self._prefix_map:
                block_id = self._prefix_map[token_hash]
                block = self._allocator._active.get(block_id) or \
                        self._allocator._lru.get(block_id)
                if block is not None:
                    self._allocator.incref(block)
                    return block, True
                # Block was evicted — fall through to allocate.
                del self._prefix_map[token_hash]

        block = self._allocator.allocate(token_hash=token_hash)
        with self._lock:
            self._prefix_map[token_hash] = block.block_id
        return block, False

    def release(self, block: KVBlock) -> None:
        """Release a block back to the pool."""
        self._allocator.decref(block)

    def usage_percent(self) -> float:
        return self._allocator.usage_percent()

    def prefix_cache_size(self) -> int:
        with self._lock:
            return len(self._prefix_map)
