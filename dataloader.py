import numpy as np
import torch
import os

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, world_size, process_rank, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size

        assert split in ["train", "val"]

        data_root = "edu_fineweb10B"
        self.shards = sorted([os.path.join(data_root, s) for s in os.listdir(data_root) if split in s])
        assert len(self.shards) > 0

        if process_rank == 0:
            print(f"found {len(self.shards)} shards for {split} split")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_idx = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.data[self.current_idx : self.current_idx + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_idx += B*T*self.world_size
        if self.current_idx + B*T*self.world_size + 1 > len(self.data):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.data = load_tokens(self.shards[self.current_shard])
            self.current_idx = self.B * self.T * self.process_rank
        return x, y
