import torch


class NaiveContiguousCache:
    def __init__(self, num_layers, num_kv_heads, head_dim, max_seq_len, device, dtype=torch.float32):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.buffers = {} 
        self.seq_lens = {}

    def allocate_request(self, request_id):
        # The defining trait of "naive": reserve the worst case immediately,
        # whether or not the request ever uses it.
        K = torch.zeros(self.num_layers, self.num_kv_heads, self.max_seq_len, self.head_dim,
                         device=self.device, dtype=self.dtype)
        V = torch.zeros_like(K)
        self.buffers[request_id] = (K, V)
        self.seq_lens[request_id] = 0

    def free_request(self, request_id):
        self.buffers.pop(request_id)
        self.seq_lens.pop(request_id)

    def append(self, request_id, layer_idx, new_k, new_v):
        pos = self.seq_lens[request_id]
        K, V = self.buffers[request_id]
        K[layer_idx, :, pos, :] = new_k[0, :, 0, :]
        V[layer_idx, :, pos, :] = new_v[0, :, 0, :]

    def gather(self, request_id, layer_idx):
        seq_len = self.seq_lens[request_id]
        if seq_len == 0:
            return None
        K, V = self.buffers[request_id]
        return K[layer_idx, :, :seq_len, :].unsqueeze(0), V[layer_idx, :, :seq_len, :].unsqueeze(0)

    def advance(self, request_id):
        self.seq_lens[request_id] += 1

    def memory_stats(self):
        bytes_per_elem = torch.tensor([], dtype=self.dtype).element_size()
        allocated_bytes = 0
        useful_bytes = 0
        per_token_bytes = self.num_kv_heads * self.head_dim * bytes_per_elem * self.num_layers * 2  # K+V
        for request_id in self.buffers:
            allocated_bytes += self.max_seq_len * per_token_bytes  # reserved, regardless of use
            useful_bytes += self.seq_lens[request_id] * per_token_bytes  # actually written
        fragmentation_pct = 100.0 * (1 - useful_bytes / allocated_bytes) if allocated_bytes else 0.0
        return {
            "allocated_bytes": allocated_bytes,
            "useful_bytes": useful_bytes,
            "fragmentation_pct": fragmentation_pct,
            "n_requests": len(self.buffers),
        }
