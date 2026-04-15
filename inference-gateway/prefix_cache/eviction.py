"""
KV-Cache Eviction Policies

Implements LRU and LFU eviction strategies for the GPU KV-cache block pool.
The BlockAllocator uses these policies when GPU memory is under pressure
and it needs to reclaim blocks from the prefix cache.

Design constraints:
  - Pinned blocks (ref_count > 0) must never be evicted.
  - Eviction must be O(1) for LRU (doubly-linked list + hash map).
  - LFU eviction uses a frequency bucket structure for O(1) min-frequency lookup.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

K = TypeVar("K")  # Block ID type (int)


@dataclass
class EvictionEntry:
    key: int
    ref_count: int = 0
    frequency: int = 0
    pinned: bool = False


class EvictionPolicy(ABC, Generic[K]):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    def access(self, key: K) -> None:
        """Record an access to key (update recency/frequency)."""

    @abstractmethod
    def insert(self, key: K) -> None:
        """Insert a new key into the eviction structure."""

    @abstractmethod
    def evict(self) -> Optional[K]:
        """
        Select and remove the best eviction candidate.
        Returns None if no evictable entry exists.
        """

    @abstractmethod
    def pin(self, key: K) -> None:
        """Mark key as pinned (must not be evicted)."""

    @abstractmethod
    def unpin(self, key: K) -> None:
        """Mark key as no longer pinned."""

    @abstractmethod
    def remove(self, key: K) -> None:
        """Forcibly remove key (e.g., after block transfer)."""


# ── LRU (Least Recently Used) ──────────────────────────────────────────────

class _DLLNode:
    """Node in a doubly-linked list."""
    __slots__ = ("key", "prev", "next")

    def __init__(self, key: int) -> None:
        self.key = key
        self.prev: Optional["_DLLNode"] = None
        self.next: Optional["_DLLNode"] = None


class LRUEviction(EvictionPolicy[int]):
    """
    O(1) LRU eviction using a doubly-linked list + hash map.
    The head of the list is the most recently used entry;
    the tail is the least recently used (eviction candidate).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._map: dict[int, _DLLNode] = {}
        self._pinned: set[int] = set()
        # Sentinel nodes.
        self._head = _DLLNode(-1)  # MRU end
        self._tail = _DLLNode(-1)  # LRU end
        self._head.next = self._tail
        self._tail.prev = self._head

    def insert(self, key: int) -> None:
        with self._lock:
            if key in self._map:
                return
            node = _DLLNode(key)
            self._map[key] = node
            self._prepend(node)

    def access(self, key: int) -> None:
        with self._lock:
            if key not in self._map:
                return
            node = self._map[key]
            self._remove_node(node)
            self._prepend(node)

    def pin(self, key: int) -> None:
        with self._lock:
            self._pinned.add(key)

    def unpin(self, key: int) -> None:
        with self._lock:
            self._pinned.discard(key)

    def evict(self) -> Optional[int]:
        with self._lock:
            # Walk from tail (LRU end) to find an unpinned entry.
            node = self._tail.prev
            while node is not self._head:
                if node.key not in self._pinned:
                    self._remove_node(node)
                    del self._map[node.key]
                    return node.key
                node = node.prev
            return None

    def remove(self, key: int) -> None:
        with self._lock:
            node = self._map.pop(key, None)
            if node:
                self._remove_node(node)
            self._pinned.discard(key)

    def _prepend(self, node: _DLLNode) -> None:
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node  # type: ignore[union-attr]
        self._head.next = node

    def _remove_node(self, node: _DLLNode) -> None:
        node.prev.next = node.next  # type: ignore[union-attr]
        node.next.prev = node.prev  # type: ignore[union-attr]

    def size(self) -> int:
        with self._lock:
            return len(self._map)


# ── LFU (Least Frequently Used) ────────────────────────────────────────────

class LFUEviction(EvictionPolicy[int]):
    """
    O(1) LFU eviction using frequency buckets.

    Maintains a min-frequency pointer to enable O(1) eviction of the
    entry with the lowest access frequency (ties broken by LRU within
    the same frequency bucket).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._freq: dict[int, int] = {}          # key → frequency
        self._buckets: dict[int, dict[int, None]] = defaultdict(dict)  # freq → ordered set of keys
        self._pinned: set[int] = set()
        self._min_freq: int = 0

    def insert(self, key: int) -> None:
        with self._lock:
            if key in self._freq:
                return
            self._freq[key] = 1
            self._buckets[1][key] = None
            if self._min_freq > 1:
                self._min_freq = 1

    def access(self, key: int) -> None:
        with self._lock:
            if key not in self._freq:
                return
            f = self._freq[key]
            self._buckets[f].pop(key, None)
            if not self._buckets[f] and f == self._min_freq:
                self._min_freq += 1
            self._freq[key] = f + 1
            self._buckets[f + 1][key] = None

    def pin(self, key: int) -> None:
        with self._lock:
            self._pinned.add(key)

    def unpin(self, key: int) -> None:
        with self._lock:
            self._pinned.discard(key)

    def evict(self) -> Optional[int]:
        with self._lock:
            freq = self._min_freq
            while freq in self._buckets:
                for key in list(self._buckets[freq]):
                    if key not in self._pinned:
                        self._buckets[freq].pop(key)
                        del self._freq[key]
                        if not self._buckets[freq]:
                            del self._buckets[freq]
                        return key
                freq += 1
            return None

    def remove(self, key: int) -> None:
        with self._lock:
            if key not in self._freq:
                return
            f = self._freq.pop(key)
            self._buckets[f].pop(key, None)
            self._pinned.discard(key)

    def size(self) -> int:
        with self._lock:
            return len(self._freq)
