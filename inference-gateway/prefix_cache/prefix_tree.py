"""
Radix-Tree Prefix Cache

Implements a prefix trie over token ID sequences. Each node represents a
block-aligned prefix (block_size tokens). Nodes store the GPU block IDs
holding the KV-cache tensors for that prefix, along with node affinity
(which physical GPU node holds those blocks), reference count, and LRU
metadata for eviction.

This enables:
  1. O(n/block_size) prefix lookup to find the longest cached prefix.
  2. Efficient sharing of KV-cache blocks across requests with common prefixes
     (e.g., a shared system prompt used by all tenants).
  3. Routing hints: the node holding the longest prefix should be preferred.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RadixNode:
    """
    A node in the radix prefix tree.

    Each node corresponds to a sequence of block_size token IDs.
    """

    # Token IDs stored at this node (length = block_size).
    token_ids: tuple[int, ...]
    # GPU block IDs for KV tensors on each node that has this prefix cached.
    # Mapping: node_name → gpu_block_id
    block_map: dict[str, int] = field(default_factory=dict)
    # Child nodes keyed by the first token ID of their segment.
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    # Reference count: number of active requests using this node's KV-cache.
    ref_count: int = 0
    # Timestamp of last access (monotonic).
    last_access: float = field(default_factory=time.monotonic)
    # Parent pointer for upward traversal during eviction.
    parent: Optional["RadixNode"] = None


@dataclass
class PrefixMatch:
    """Result of a prefix tree lookup."""

    # Number of full blocks matched.
    num_matched_blocks: int
    # Leaf node of the longest matching prefix.
    node: Optional[RadixNode]
    # The GPU node name that holds the most matched blocks.
    preferred_node: str
    # Fraction of input tokens covered by the cached prefix.
    hit_rate: float


class RadixPrefixTree:
    """
    Thread-safe radix tree for KV-cache prefix matching.

    Token ID sequences are split into blocks of block_size tokens. Each
    block is a tree level. Lookup finds the deepest node that matches a
    prefix of the input token IDs.
    """

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = block_size
        self._lock = threading.Lock()
        # Sentinel root node (empty token_ids).
        self._root = RadixNode(token_ids=())
        self._num_nodes = 0

    # ── Insertion ──────────────────────────────────────────────────────────

    def insert(
        self, token_ids: list[int], node_name: str, gpu_block_id: int
    ) -> None:
        """
        Insert a token sequence into the tree, recording that node_name
        holds the KV-cache for this prefix at gpu_block_id.

        Token IDs are split into blocks of block_size. For each block a
        tree node is created (or found) and the block_map is updated.
        """
        with self._lock:
            current = self._root
            for block_start in range(0, len(token_ids), self.block_size):
                block = tuple(token_ids[block_start : block_start + self.block_size])
                if len(block) < self.block_size:
                    # Partial block — only insert complete blocks.
                    break
                first_token = block[0]
                if first_token not in current.children:
                    node = RadixNode(token_ids=block, parent=current)
                    current.children[first_token] = node
                    self._num_nodes += 1
                else:
                    node = current.children[first_token]
                node.block_map[node_name] = gpu_block_id
                node.last_access = time.monotonic()
                current = node

    # ── Lookup ─────────────────────────────────────────────────────────────

    def match_prefix(self, token_ids: list[int]) -> PrefixMatch:
        """
        Find the longest prefix of token_ids that is in the tree.

        Returns a PrefixMatch with the number of matched blocks, the deepest
        matched node, and the preferred GPU node (highest block count).
        """
        with self._lock:
            current = self._root
            matched_blocks = 0
            last_node: Optional[RadixNode] = None
            node_block_counts: dict[str, int] = {}

            for block_start in range(0, len(token_ids), self.block_size):
                block = tuple(token_ids[block_start : block_start + self.block_size])
                if len(block) < self.block_size:
                    break
                first_token = block[0]
                child = current.children.get(first_token)
                if child is None or child.token_ids != block:
                    break
                matched_blocks += 1
                last_node = child
                child.last_access = time.monotonic()
                for node_name in child.block_map:
                    node_block_counts[node_name] = (
                        node_block_counts.get(node_name, 0) + 1
                    )
                current = child

            preferred_node = max(node_block_counts, key=node_block_counts.get) \
                if node_block_counts else ""

            total_blocks = len(token_ids) // self.block_size
            hit_rate = matched_blocks / total_blocks if total_blocks > 0 else 0.0

            return PrefixMatch(
                num_matched_blocks=matched_blocks,
                node=last_node,
                preferred_node=preferred_node,
                hit_rate=hit_rate,
            )

    # ── Reference counting ─────────────────────────────────────────────────

    def incref(self, node: RadixNode) -> None:
        """Pin a node's KV-cache blocks (active request is using them)."""
        with self._lock:
            current: Optional[RadixNode] = node
            while current is not None and current is not self._root:
                current.ref_count += 1
                current = current.parent

    def decref(self, node: RadixNode) -> None:
        """Unpin a node's KV-cache blocks."""
        with self._lock:
            current: Optional[RadixNode] = node
            while current is not None and current is not self._root:
                current.ref_count = max(0, current.ref_count - 1)
                current = current.parent

    # ── Eviction ───────────────────────────────────────────────────────────

    def evict_lru(self, max_nodes: int = 10) -> list[RadixNode]:
        """
        Evict up to max_nodes leaf nodes in LRU order (oldest last_access,
        ref_count == 0).

        Returns the list of evicted nodes so the caller can free GPU blocks.
        """
        with self._lock:
            candidates = self._collect_evictable_leaves(self._root)
            candidates.sort(key=lambda n: n.last_access)
            evicted = []
            for node in candidates[:max_nodes]:
                if node.parent is not None:
                    first_token = node.token_ids[0]
                    node.parent.children.pop(first_token, None)
                evicted.append(node)
                self._num_nodes -= 1
            return evicted

    def _collect_evictable_leaves(self, node: RadixNode) -> list[RadixNode]:
        """Recursively collect leaf nodes with ref_count == 0."""
        if not node.children:
            if node is not self._root and node.ref_count == 0:
                return [node]
            return []
        evictable = []
        for child in list(node.children.values()):
            evictable.extend(self._collect_evictable_leaves(child))
        return evictable

    # ── Stats ───────────────────────────────────────────────────────────────

    def size(self) -> int:
        return self._num_nodes

    def depth(self) -> int:
        """Return the maximum tree depth (number of block levels)."""
        with self._lock:
            return self._max_depth(self._root, 0)

    def _max_depth(self, node: RadixNode, d: int) -> int:
        if not node.children:
            return d
        return max(self._max_depth(c, d + 1) for c in node.children.values())
