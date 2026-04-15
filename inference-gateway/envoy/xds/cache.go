package main

import (
	"sync"
	"sync/atomic"

	"github.com/envoyproxy/go-control-plane/pkg/cache/v3"
	"go.uber.org/zap"
)

// VersionedCache wraps the xDS snapshot cache with an auto-incrementing version counter
// and provides thread-safe snapshot retrieval for metrics and debugging.
type VersionedCache struct {
	inner   cache.SnapshotCache
	version atomic.Int64
	log     *zap.Logger

	mu        sync.RWMutex
	nodeStats map[string]nodeCacheStats
}

// nodeCacheStats tracks per-node snapshot delivery stats.
type nodeCacheStats struct {
	SnapshotVersion string
	LastPushTime    int64 // unix nano
	PushCount       int64
}

// NewVersionedCache wraps a SnapshotCache.
func NewVersionedCache(inner cache.SnapshotCache, log *zap.Logger) *VersionedCache {
	return &VersionedCache{
		inner:     inner,
		log:       log,
		nodeStats: make(map[string]nodeCacheStats),
	}
}

// NextVersion atomically increments and returns the next snapshot version string.
func (vc *VersionedCache) NextVersion() string {
	v := vc.version.Add(1)
	return fmt.Sprintf("%d", v)
}

// Inner returns the underlying snapshot cache.
func (vc *VersionedCache) Inner() cache.SnapshotCache {
	return vc.inner
}

// RecordPush records a snapshot push event for the given node group.
func (vc *VersionedCache) RecordPush(nodeGroup, version string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	st := vc.nodeStats[nodeGroup]
	st.SnapshotVersion = version
	st.LastPushTime = time.Now().UnixNano()
	st.PushCount++
	vc.nodeStats[nodeGroup] = st
}

// Stats returns a copy of per-node snapshot stats.
func (vc *VersionedCache) Stats() map[string]nodeCacheStats {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	out := make(map[string]nodeCacheStats, len(vc.nodeStats))
	for k, v := range vc.nodeStats {
		out[k] = v
	}
	return out
}
