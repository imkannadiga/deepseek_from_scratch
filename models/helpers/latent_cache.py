import torch

class MLADynamicCache:
    def __init__(self):
        self.c_kv_cache = []   # one compressed latent per layer

    def update(self, layer_idx, new_c_kv):
        if layer_idx >= len(self.c_kv_cache):
            self.c_kv_cache.append(new_c_kv)
        else:
            self.c_kv_cache[layer_idx] = torch.cat(
                [self.c_kv_cache[layer_idx], new_c_kv], dim=1
            )
        return self.c_kv_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        # Must be per-layer: mid-forward, layer 0 has already appended this
        # step's token while later layers have not, so reading layer 0 from
        # layer 3 would report a length one too long.
        if len(self.c_kv_cache) <= layer_idx:
            return 0
        return self.c_kv_cache[layer_idx].shape[1]

    def size_bytes(self):
        return sum(t.nbytes for t in self.c_kv_cache)

    def clear(self):
        self.c_kv_cache = []