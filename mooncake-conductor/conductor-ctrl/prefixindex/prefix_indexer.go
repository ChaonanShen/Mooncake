package prefixindex

import (
	"encoding/binary"
	"fmt"
	"log/slog"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"conductor/common"

	"github.com/cespare/xxhash/v2"
)

var (
	enableCpuEviction   = common.LoadBoolEnv("ENABLE_CPU_EVICTION", false)
	maxCpuKeyNum        = int64(common.LoadIntEnv("MAX_CPU_KEY_NUM", 30000))
	cpuKeyEvictionRatio = common.LoadFloatEnv("CPU_KEY_EVICTION_RATIO", 0.2)
	// TODO 
	// The eviction parameter here is used to control that the number of CPU cache prefixes cannot grow without limit.
	// The best way is to add an zmq publisher in mooncake-store to actively notify conductor of the block eviction.
)

type ModelContext struct {
	ModelName string
	LoraName  string // None represents no LoRA adapter
	BlockSize int64
	// TODO @yejj710
	// Confirm the difference between the previously discussed cache_salt and additionalSalt.
	//The current understanding is that cache_salt is used to ensure data isolation between different customers,
	// and it seems it can be directly added to additionalSalt.
	AdditionalSalt string
	TenantID       string
	InstanceID     string // unique identifier for each API server
}

type CacheStoreInfo struct {
	// TODO  Currently, the KV cache at different levels is not distinguished.
	// In the future, the caches of Mooncake and inference engines (vLLM, SGLang)
	// should be handled separately.
	engineLastAccessTime atomic.Int64
	TotalReplicaNums     atomic.Int64
	mediumSet            map[string]struct{}
	dpRankSet            map[int64]struct{} // indicate the dp_rank that the block is cached on
	// LRU linked list pointers
	lruPrev *CacheStoreInfo
	lruNext *CacheStoreInfo
}

type HashMapStore struct {
	// conductor prefixHash -> cachestore
	prefixMap map[uint64]*CacheStoreInfo

	createTime    time.Time
	lastAccess    atomic.Int64
	totalPrefixes int64
	// LRU linked list: head is least recently used, tail is most recently used
	lruHead *CacheStoreInfo
	lruTail *CacheStoreInfo
}

type ContextData struct {
	prefixMu  sync.RWMutex
	hashmapMu sync.RWMutex

	prefixStore *HashMapStore
	seed        uint64

	DpSize map[int64]struct{}

	proxyHashMapping map[uint64]uint64 // engine block hash -> conductor prefix hash
}

type PrefixCacheTable struct {
	// TODO use instance_id to distinguish different engine instances
	contextMap sync.Map // ModelContext → *ContextData

	contextCount atomic.Int32
}

type CacheHitResult struct {
	LongestMatchTokens int64           `json:"longest_matched"`
	DP                 map[int64]int64 `json:"DP"`
	GPU                int64           `json:"GPU"`
	CPU                int64           `json:"CPU"`
	DISK               int64           `json:"DISK"`
}

type ModelContextView struct {
	ModelName      string `json:"model_name"`
	LoraName       string `json:"lora_name"`
	BlockSize      int64  `json:"block_size"`
	AdditionalSalt string `json:"additional_salt"`
	TenantID       string `json:"tenant_id"`
	InstanceID     string `json:"instance_id"`
}

type GlobalView struct {
	ContextCount  int32               `json:"context_count"`
	ModelContexts []ModelContextView  `json:"model_contexts"`
	ProxyHashMap  []map[uint64]uint64 `json:"hashmap"`
}

func GenerateSeedFromEnv() uint64 {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	envSeed := common.LoadIntEnv("CONDUCTOR_SEED", -1)

	var seed uint64
	if envSeed != -1 {
		seed = uint64(envSeed)
	} else {
		seed = r.Uint64()
	}
	return seed
}

func NewPrefixCacheTable() *PrefixCacheTable {
	p := &PrefixCacheTable{}
	return p
}

func (p *PrefixCacheTable) getContextData(modelcontext *ModelContext) *ContextData {
	ctx_value := *modelcontext
	value, exists := p.contextMap.Load(ctx_value)
	if exists {
		return value.(*ContextData)
	}
	seedValue := xxhash.Sum64String(modelcontext.AdditionalSalt)
	newContextData := &ContextData{
		prefixStore: &HashMapStore{
			prefixMap:     make(map[uint64]*CacheStoreInfo),
			createTime:    time.Now(),
			totalPrefixes: 0,
		},
		proxyHashMapping: make(map[uint64]uint64),
		seed:             seedValue,
		DpSize:           make(map[int64]struct{}),
	}
	newContextData.prefixStore.lastAccess.Store(time.Now().Unix())
	p.contextMap.Store(ctx_value, newContextData)
	slog.Debug("in getContextData", "modelcontext", modelcontext)
	p.contextCount.Add(1)
	return newContextData
}

func (p *PrefixCacheTable) AddDpSize(modelcontext *ModelContext, dpRank int64) {
	// value, exists := p.contextMap.Load(modelcontext)
	contextData := p.getContextData(modelcontext)
	contextData.DpSize[dpRank] = struct{}{}
}

func (p *PrefixCacheTable) ComputePrefixHash(modelcontext *ModelContext, tokenIds []int32, cacheSalt uint64) []uint64 {
	// cacheSalt is used to seperate hash from different customers
	numBlocks := len(tokenIds) / int(modelcontext.BlockSize)
	prefixHashes := make([]uint64, 0, numBlocks)

	var parentHash uint64 = cacheSalt

	for i := 0; i < numBlocks; i++ {
		start := i * int(modelcontext.BlockSize)
		end := start + int(modelcontext.BlockSize)
		if end > len(tokenIds) {
			break
		}
		hashValue := p.computeHash(parentHash, tokenIds[start:end])
		prefixHashes = append(prefixHashes, hashValue)
		parentHash = hashValue
	}
	return prefixHashes
}

func (p *PrefixCacheTable) CacheHitCompute(modelcontext *ModelContext, tokenIds []int32) *CacheHitResult {
	value, exists := p.contextMap.Load(*modelcontext)
	prefixMatchResult := &CacheHitResult{
		LongestMatchTokens: 0,
		DP:                 map[int64]int64{},
		GPU:                0,
		CPU:                0,
		DISK:               0,
	}

	if !exists {
		slog.Error("In CacheHitCompute, contextData not found")
		return prefixMatchResult
	}
	contextData := value.(*ContextData)
	cacheSalt := xxhash.Sum64String(modelcontext.AdditionalSalt)

	prefixHashes := p.ComputePrefixHash(modelcontext, tokenIds, cacheSalt)

	// TODO @yejj710
	// When there is no data in contextData, what information should be returned for the matched modelcontext
	// This is related to function `AddDpSize`

	slog.Debug("In CacheHitCompute", "prefixHashes", prefixHashes)

	contextData.prefixMu.RLock()
	defer contextData.prefixMu.RUnlock()
	prefixStore := contextData.prefixStore

	// reserve prefixHashes and then compute cache hit
	for _, prefixHash := range prefixHashes {
		cacheStoreInfo, exists := prefixStore.prefixMap[prefixHash]
		slog.Debug("In CacheHitCompute", "cacheStoreInfo", cacheStoreInfo)
		// chained hash, break if no replica exists
		if !exists || cacheStoreInfo.TotalReplicaNums.Load() == 0 {
			break
		}

		cacheHit := false

		for key := range cacheStoreInfo.mediumSet {
			if key == "cpu" {
				prefixMatchResult.CPU += modelcontext.BlockSize
				cacheHit = true
			} else if key == "GPU" {
				prefixMatchResult.GPU += modelcontext.BlockSize
				cacheHit = true
			} else {
				slog.Warn("In CacheHitCompute, unknown medium type", "medium", key)
			}

		}
		if cacheHit {
			prefixMatchResult.LongestMatchTokens += modelcontext.BlockSize
			for dpRank := range cacheStoreInfo.dpRankSet {
				prefixMatchResult.DP[dpRank] += modelcontext.BlockSize
			}
			cacheStoreInfo.engineLastAccessTime.Store(time.Now().Unix())
			// move to tail (most recently used)
			prefixStore.addToLRUTail(cacheStoreInfo)
			// TODO update LRU list asynchronously
		}
	}

	prefixStore.lastAccess.Store(time.Now().Unix())

	return prefixMatchResult
}

func (p *PrefixCacheTable) ProcessStoreEvent(event common.StoredEvent, dpRank int64) error {
	if len(event.BlockHashes) == 0 {
		return nil
	}
	tenantID := "default"

	slog.Debug("In ProcessStoreEvent", "modelName", event.ModelName, "instanceID", event.InstanceID, "dpRank", dpRank)
	contextData := p.getContextData(&ModelContext{
		ModelName:      event.ModelName,
		LoraName:       event.LoraName,
		BlockSize:      event.BlockSize,
		TenantID:       tenantID,
		AdditionalSalt: "",
		InstanceID:     event.InstanceID,
	})

	contextData.hashmapMu.Lock()
	defer contextData.hashmapMu.Unlock()
	proxyHashMap := contextData.proxyHashMapping

	if len(event.BlockHashes)*int(event.BlockSize) != len(event.TokenIds) {
		if len(event.BlockHashes) != 1 {
			return fmt.Errorf("block hashes and tokens length mismatch")
		}
	}

	newPrefixStore := make([]struct {
		hashValue uint64
		engineID  string
	}, 0)

	var parentHash uint64 = contextData.seed

	// TODO If the ParentBlockHash happens to be 0, a bug will occur here, because 0 is a valid hash value.
	if event.ParentBlockHash != 0 {
		slog.Debug("parent Block HASH is not None.")
		if pbh, exists := proxyHashMap[event.ParentBlockHash]; exists {
			parentHash = pbh
		}
	}

	for i, blockHash := range event.BlockHashes {
		// cache already exists, add engine info and continue
		if existingHash, exists := proxyHashMap[blockHash]; exists {
			newPrefixStore = append(newPrefixStore, struct {
				hashValue uint64
				engineID  string
			}{existingHash, event.InstanceID})
			continue
		}
		// if not exists, compute hash
		hashValue := p.computeHash(parentHash, event.TokenIds[i*int(event.BlockSize):(i+1)*int(event.BlockSize)])
		parentHash = hashValue

		proxyHashMap[blockHash] = hashValue

		newPrefixStore = append(newPrefixStore, struct {
			hashValue uint64
			engineID  string
		}{
			hashValue: hashValue,
			engineID:  event.InstanceID,
		})

	}
	if len(newPrefixStore) > 0 {
		contextData.prefixMu.Lock()
		defer contextData.prefixMu.Unlock()

		prefixStore := contextData.prefixStore
		for _, newPrefix := range newPrefixStore {
			slog.Debug("show new prefix data", "newPrefix", newPrefix)
			p.addNewPrefixStore(prefixStore, newPrefix.hashValue, newPrefix.engineID, event.Medium, dpRank)
		}
		if enableCpuEviction && prefixStore.totalPrefixes > maxCpuKeyNum {
			evictedCount := prefixStore.evictLRU(cpuKeyEvictionRatio)
			slog.Info("LRU eviction triggered", "totalPrefixes", prefixStore.totalPrefixes, "evictedCount", evictedCount)
		}

	}

	return nil
}

func (p *PrefixCacheTable) ProcessRemoveEvent(event common.RemovedEvent, dpRank int64, instanceID string) error {
	if len(event.BlockHashes) == 0 {
		return nil
	}

	contextData := p.getContextData(&ModelContext{
		ModelName:      event.ModelName,
		LoraName:       event.LoraName,
		BlockSize:      event.BlockSize,
		TenantID:       "default",
		AdditionalSalt: "",
		InstanceID:     instanceID,
	})

	contextData.hashmapMu.Lock()
	defer contextData.hashmapMu.Unlock()
	proxyHashMap := contextData.proxyHashMapping
	removeConductorHash := make([]uint64, 0, len(event.BlockHashes))

	// delete proxyHashMapping
	contextData.prefixMu.Lock()
	defer contextData.prefixMu.Unlock()
	prefixStore := contextData.prefixStore
	for _, blockHash := range event.BlockHashes {
		if conductorHash, exists := proxyHashMap[blockHash]; exists {
			removeConductorHash = append(removeConductorHash, conductorHash)
			// Only delete proxyHashMapping entry when all replicas are removed
			cacheStoreInfo, exists := prefixStore.prefixMap[conductorHash]
			if !exists {
				continue
			}
			if cacheStoreInfo.TotalReplicaNums.Load() == 1 {
				delete(proxyHashMap, blockHash)
			}
		}
	}

	// update prefixStore
	for _, conductorHash := range removeConductorHash {
		cacheStoreInfo, exists := prefixStore.prefixMap[conductorHash]
		if !exists {
			continue
		}

		cacheStoreInfo.TotalReplicaNums.Add(-1)

		// Remove per-instance metadata
		delete(cacheStoreInfo.dpRankSet, dpRank)
		delete(cacheStoreInfo.mediumSet, event.Medium)

		// Only delete entry when all replicas are removed
		if cacheStoreInfo.TotalReplicaNums.Load() <= 0 {
			delete(prefixStore.prefixMap, conductorHash)
			prefixStore.totalPrefixes--
		}
	}

	return nil
}

func (p *PrefixCacheTable) computeHash(parentHash uint64, blockTokenIDs []int32) uint64 {
	// digest := xxhash.NewWithSeed()
	digest := xxhash.New()
	var parentHashBytes [8]byte
	binary.LittleEndian.PutUint64(parentHashBytes[:], parentHash)
	_, _ = digest.Write(parentHashBytes[:])

	var tokenIDsBytes [8]byte
	for _, tokenID := range blockTokenIDs {
		binary.LittleEndian.PutUint32(tokenIDsBytes[:], uint32(tokenID))
		_, _ = digest.Write(tokenIDsBytes[:])
	}
	return digest.Sum64()
}

func (p *PrefixCacheTable) addNewPrefixStore(prefixStore *HashMapStore, hashValue uint64, instanceID string, medium string, dpRank int64) {
	now := time.Now().Unix()
	if prefixStore.prefixMap[hashValue] == nil {
		slog.Debug("in addNewPrefixStore, prefixStore.prefixMap[hashValue] is nil", "hashValue", hashValue)
		prefixStore.prefixMap[hashValue] = &CacheStoreInfo{
			mediumSet:            make(map[string]struct{}),
			dpRankSet:            make(map[int64]struct{}),
		}
		prefixStore.totalPrefixes++
	}
	cacheStoreInfo := prefixStore.prefixMap[hashValue]

	cacheStoreInfo.engineLastAccessTime.Store(now)
	cacheStoreInfo.TotalReplicaNums.Add(1)
	// TODO If using Mooncake-Store, you do not need to set dpRank, because it does not distinguish between kv-blocks of different dpRanks.
	cacheStoreInfo.mediumSet[medium] = struct{}{}
	cacheStoreInfo.dpRankSet[dpRank] = struct{}{}
	slog.Debug("in addNewPrefixStore", "conductor_hash", hashValue, "current_mediumset", cacheStoreInfo.mediumSet[medium])

	// move to tail (most recently used)
	prefixStore.addToLRUTail(cacheStoreInfo)
}

func (p *PrefixCacheTable) GetGlobalView() *GlobalView {
	view := &GlobalView{
		ContextCount:  p.contextCount.Load(),
		ModelContexts: make([]ModelContextView, 0),
		ProxyHashMap:  make([]map[uint64]uint64, 0),
	}

	p.contextMap.Range(func(key, value interface{}) bool {
		ctx := key.(ModelContext)
		contextData := value.(*ContextData)

		ctxView := ModelContextView{
			ModelName:      ctx.ModelName,
			LoraName:       ctx.LoraName,
			BlockSize:      ctx.BlockSize,
			AdditionalSalt: ctx.AdditionalSalt,
			TenantID:       ctx.TenantID,
			InstanceID:     ctx.InstanceID,
		}

		contextData.prefixMu.RLock()
		defer contextData.prefixMu.RUnlock()

		contextData.hashmapMu.RLock()
		defer contextData.hashmapMu.RUnlock()

		view.ProxyHashMap = append(view.ProxyHashMap, contextData.proxyHashMapping)
		view.ModelContexts = append(view.ModelContexts, ctxView)

		return true
	})

	return view
}

// move or add a CacheStoreInfo to the tail of the LRU list (most recently used)
func (h *HashMapStore) addToLRUTail(cacheNode *CacheStoreInfo) {
	if cacheNode == nil {
		return
	}

	// If already in list, remove it first
	if cacheNode.lruPrev != nil || cacheNode.lruNext != nil || h.lruHead == cacheNode {
		h.removeFromLRU(cacheNode)
	}

	if h.lruTail == nil {
		// Empty list
		h.lruHead = cacheNode
		h.lruTail = cacheNode
		cacheNode.lruPrev = nil
		cacheNode.lruNext = nil
	} else {
		h.lruTail.lruNext = cacheNode
		cacheNode.lruPrev = h.lruTail
		cacheNode.lruNext = nil
		h.lruTail = cacheNode
	}
}

// remove a CacheStoreInfo from the LRU list
func (h *HashMapStore) removeFromLRU(cacheNode *CacheStoreInfo) {
	if cacheNode == nil {
		return
	}

	if cacheNode.lruPrev != nil {
		cacheNode.lruPrev.lruNext = cacheNode.lruNext
	} else {
		// cacheNode is head
		h.lruHead = cacheNode.lruNext
	}

	if cacheNode.lruNext != nil {
		cacheNode.lruNext.lruPrev = cacheNode.lruPrev
	} else {
		// cacheNode is tail
		h.lruTail = cacheNode.lruPrev
	}

	cacheNode.lruPrev = nil
	cacheNode.lruNext = nil
}

// evict the least recently used cpu mediums based on eviction ratio
func (h *HashMapStore) evictLRU(ratio float64) int {
	if h.lruHead == nil || ratio <= 0 {
		return 0
	}
	// TODO Here it is assumed that each CacheStoreInfo contains a cpu medium, but in practice such assumptions should not be made. 
	// Instead, a separate variable should be used to record the number of cpu mediums.
	targetCount := int(float64(h.totalPrefixes) * ratio)
	evictedCount := 0

	// Traverse from head (least recently used)
	for h.lruHead != nil && evictedCount < targetCount {
		cacheNode := h.lruHead

		// Only evict `cpu` medium
		if _, exist := cacheNode.mediumSet["cpu"]; exist {
			delete(cacheNode.mediumSet, "cpu")

			// If no mediums left, remove the entry entirely
			if len(cacheNode.mediumSet) == 0 {
				h.removeFromLRU(cacheNode)
				h.deleteCacheStoreInfo(cacheNode)
				// TODO remove proxyHashMapping in ContextData, 
				// currently we do not maintain reverse mapping from conductor hash to proxy hash, 
				// which makes it hard to delete the proxyHashMapping entry. 
				h.totalPrefixes--
			}
		} else {
			// Move to LRU-List tail
			h.removeFromLRU(cacheNode)
			h.addToLRUTail(cacheNode)
		}
		evictedCount++
	}

	return evictedCount
}


func (h *HashMapStore) deleteCacheStoreInfo(cacheNode *CacheStoreInfo) {

	for k, v := range h.prefixMap {
		if v == cacheNode {
			delete(h.prefixMap, k)
			return
		}
	}
}