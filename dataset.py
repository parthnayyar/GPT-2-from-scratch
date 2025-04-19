from typing import Literal
import numpy as np
import torch

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, B: int, T: int, world_size: int, process_rank: int, split: Literal["train", "val"], dataset: Literal["shakespeare", "fineweb"]="fineweb"):
        super().__init__()
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size

        assert split in ["train", "val"]
        assert dataset in ["shakespeare", "fineweb"]

        self.data = np.memmap(f"{split}.bin", dtype=np.uint16, mode="r") if dataset == "fineweb" else np.load(f"shakespeare_{split}.npy")
    
    def __len__(self):
        B, T = self.B, self.T
        quotient = (len(self.data)-1)//(B*T*self.world_size)
        remainder = (len(self.data)-1)%(B*T*self.world_size)
        return quotient+1 if (self.process_rank+1)*B*T <= remainder else quotient
    
    def __getitem__(self, idx):
        B, T = self.B, self.T
        buf = torch.tensor(self.data[idx*B*T*self.world_size + self.process_rank*B*T : idx*B*T*self.world_size + (self.process_rank+1)*B*T + 1], dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        return x, y
    