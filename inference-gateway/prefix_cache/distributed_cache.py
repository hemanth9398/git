"""
Distributed Prefix Cache

Maintains shared prefix-cache metadata across all vLLM nodes using Redis
as the coordination layer. The actual KV tensors remain GPU-local; Redis
stores only the mapping from prefix hash → {node_name, block_id, hit_count}.

This enables the routing engine to ask:
  "Which node has the longest cached prefix for this request?"
without inspecting each node individually.

Architecture:
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  vLLM Node-0  │    │  vLLM Node-1  │    │  vLLM Node-2  │
  │  (prefill)   │    │  (decode)    │    │  (decode)    │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
         │ publish           │ publish           │ publish
         ▼                   ▼                   ▼
  ┌──────────────────────────────────────────────────────┐
  │           Redis Cluster (hash slots, 6 shards)        │
  │   key: prefix:<hash64>  value: {node, block_id, ttl}  │
  └──────────────────────────────────────────────────────┘
         ▲
         │ query (routing engine / ext_proc)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger("distributed-cache")

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis-cluster.inference-gateway.svc.cluster.local:6379")
PREFIX_TTL_SECONDS = int(os.environ.get("PREFIX_CACHE_TTL_SECONDS", "3600"))
KEY_PREFIX = "prefix:"


@dataclass
class PrefixEntry:
    """Metadata for a cached prefix stored in Redis."""

    node_name: str
    gpu_block_id: int
    token_count: int
    hit_count: int = 0
    created_at: float = 0.0
    last_access: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "PrefixEntry":
        return cls(**json.loads(data))


class DistributedPrefixCache:
    """
    Redis-backed distributed prefix cache registry.

    Each vLLM node calls register() when it computes a new KV-cache prefix.
    The routing engine calls lookup() to find which node holds a given prefix.
    """

    def __init__(self, redis_url: str = REDIS_URL, ttl: int = PREFIX_TTL_SECONDS) -> None:
        self._redis: Optional[aioredis.Redis] = None
        self._redis_url = redis_url
        self._ttl = ttl

    async def connect(self) -> None:
        self._redis = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=1,
            retry_on_timeout=True,
            max_connections=50,
        )
        await self._redis.ping()
        logger.info("Connected to Redis: %s", self._redis_url)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    # ── Write operations ────────────────────────────────────────────────────

    async def register(
        self, prefix_hash: int, node_name: str, gpu_block_id: int, token_count: int
    ) -> None:
        """
        Register that node_name has cached the prefix identified by prefix_hash.
        If multiple nodes hold the same prefix, we store the most recent entry
        (SETNX semantics — first writer wins, refresh on TTL expiry).
        """
        if self._redis is None:
            return
        key = f"{KEY_PREFIX}{prefix_hash}"
        entry = PrefixEntry(
            node_name=node_name,
            gpu_block_id=gpu_block_id,
            token_count=token_count,
            created_at=time.time(),
            last_access=time.time(),
        )
        try:
            # Use SET NX (only set if not exists) to avoid overwriting a fresher entry.
            await self._redis.set(key, entry.to_json(), ex=self._ttl, nx=True)
        except Exception as exc:
            logger.warning("Redis register failed for hash %d: %s", prefix_hash, exc)

    async def touch(self, prefix_hash: int) -> None:
        """Refresh the TTL of a cached prefix (called on cache hit)."""
        if self._redis is None:
            return
        key = f"{KEY_PREFIX}{prefix_hash}"
        try:
            pipe = self._redis.pipeline()
            await pipe.expire(key, self._ttl)
            await pipe.hincrby(key, "hit_count", 1)
            await pipe.execute()
        except Exception as exc:
            logger.debug("Redis touch failed for hash %d: %s", prefix_hash, exc)

    async def invalidate(self, prefix_hash: int) -> None:
        """Remove a prefix entry (e.g., block was evicted from GPU)."""
        if self._redis is None:
            return
        try:
            await self._redis.delete(f"{KEY_PREFIX}{prefix_hash}")
        except Exception as exc:
            logger.warning("Redis invalidate failed for hash %d: %s", prefix_hash, exc)

    # ── Read operations ─────────────────────────────────────────────────────

    async def lookup(self, prefix_hash: int) -> Optional[PrefixEntry]:
        """
        Look up which node holds the KV-cache for prefix_hash.
        Returns None on miss.
        """
        if self._redis is None:
            return None
        key = f"{KEY_PREFIX}{prefix_hash}"
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            return PrefixEntry.from_json(data)
        except Exception as exc:
            logger.warning("Redis lookup failed for hash %d: %s", prefix_hash, exc)
            return None

    async def bulk_lookup(self, hashes: list[int]) -> dict[int, Optional[PrefixEntry]]:
        """
        Batch lookup for multiple prefix hashes using a Redis pipeline.
        Returns a mapping of hash → entry (None on miss).
        """
        if self._redis is None or not hashes:
            return {}
        keys = [f"{KEY_PREFIX}{h}" for h in hashes]
        try:
            values = await self._redis.mget(*keys)
            return {
                h: PrefixEntry.from_json(v) if v else None
                for h, v in zip(hashes, values)
            }
        except Exception as exc:
            logger.warning("Redis bulk_lookup failed: %s", exc)
            return {}

    async def node_prefix_count(self, node_name: str) -> int:
        """Return the number of cached prefixes attributed to node_name."""
        if self._redis is None:
            return 0
        try:
            count = 0
            async for key in self._redis.scan_iter(f"{KEY_PREFIX}*", count=100):
                data = await self._redis.get(key)
                if data:
                    entry = PrefixEntry.from_json(data)
                    if entry.node_name == node_name:
                        count += 1
            return count
        except Exception as exc:
            logger.warning("Redis node_prefix_count failed: %s", exc)
            return 0
