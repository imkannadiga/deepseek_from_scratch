import torch

class DynamicCache:
    def __init__(self):
        self.key_cache   = [] 
        self.value_cache = [] 

    def update(self, layer_idx, new_k, new_v):
        if layer_idx >= len(self.key_cache):
            # first time this layer is seen -- just store
            self.key_cache.append(new_k)
            self.value_cache.append(new_v)
        else:
            # append new token's K,V to the running sequence
            self.key_cache[layer_idx]   = torch.cat([self.key_cache[layer_idx],   new_k], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], new_v], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def size_bytes(self):
        return sum(t.nbytes for t in self.key_cache) + sum(t.nbytes for t in self.value_cache)

    def clear(self):
        self.key_cache = []
        self.value_cache = []


def make_cache(model):
    # MLA caches one compressed latent per layer, everything else caches K and V
    from models.attention.rope_mla import RopeMLA
    from models.helpers.latent_cache import MLADynamicCache

    for m in model.modules():
        if isinstance(m, RopeMLA):
            return MLADynamicCache()
    return DynamicCache()