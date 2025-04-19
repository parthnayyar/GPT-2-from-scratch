from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import tiktoken
import inspect
import torch
import math
import time
import os


ddp = int(os.environ.get("RANK", -1)) != -1 
if ddp:
    assert torch.cuda.is_available()
    torch.distributed.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


@dataclass
class GPTConfig:
    context_window: int = 1024 # context window
    vocab_size: int = 50257 # number of tokens (50000 BPE merges + 256 byte tokens + 1 <|endoftext|> token)
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class CasualSelfAttention(torch.nn.Module): 
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn: torch.nn.Linear = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj: torch.nn.Linear = torch.nn.Linear(config.n_embd, config.n_embd)
        self.STD_SCALE_INIT: bool = True
        self.n_head: int = config.n_head
        self.n_embd: int = config.n_embd
        self.register_buffer(
            "bias", 
            torch.tril(
                torch.ones(config.context_window, config.context_window)
            ).view(
                1, 
                1, 
                config.context_window,
                config.context_window
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5)) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = torch.nn.functional.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y



class MLP(torch.nn.Module): 
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc: torch.nn.Linear = torch.nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu: torch.nn.GELU = torch.nn.GELU(approximate="tanh")
        self.c_proj: torch.nn.Linear = torch.nn.Linear(4 * config.n_embd, config.n_embd)
        self.STD_SCALE_INIT: bool = True

    def forward(self, x) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(torch.nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1: torch.nn.LayerNorm = torch.nn.LayerNorm(config.n_embd)
        self.attn: CasualSelfAttention = CasualSelfAttention(config)
        self.ln_2: torch.nn.LayerNorm = torch.nn.LayerNorm(config.n_embd)
        self.mlp: MLP = MLP(config)

    def forward(self, x) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(torch.nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config: GPTConfig = config

        self.transformer: Iterable[torch.nn.Module] = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe = torch.nn.Embedding(config.context_window, config.n_embd),
            h = torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = torch.nn.LayerNorm(config.n_embd,),
        ))
        self.lm_head: torch.nn.Linear = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme (saves aounr 40M / 30% of parameters)
        self.transformer.wte.weight = self.lm_head.weight 

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            std = 0.02
            if hasattr(module, "STD_SCALE_INIT"):
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.context_window, f"Cannot forward sequence of length {T}, block size is only {self.config.context_window}"
        
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),

            )
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() > 1]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and "cuda" in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type) -> GPT2:
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        from transformers import GPT2LMHeadModel
        print(f"loading pretrained {model_type} from huggingface")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768), # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600), # 1558M
        }[model_type]
        config_args["context_window"] = 1024
        config_args["vocab_size"] = 50257
        config = GPTConfig(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # discard buffer

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias") or k.endswith(".attn.bias")] # discard buffer
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "attn.c_attn.bias", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

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


total_batch_size = 524288 # (closest power of 2 to 0.5mil, 2**19)
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B*T*ddp_world_size) == 0
grad_accumulation_steps = total_batch_size // (B*T*ddp_world_size) 
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"grad accumulation steps: {grad_accumulation_steps}")


torch.set_float32_matmul_precision("high")

model: GPT2 = GPT2(GPTConfig(vocab_size=50304))
model.to(DEVICE)
model = torch.compile(model)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 375e6 / 2**19 as per GPT-3
n_epochs = 1
max_steps = n_epochs * 19073 # 10e9 / 2**19 for 1 epoch (10e9 tokens, total_batch_size = 2**19 = tokens processed per step)
def get_lr(i):
    if i < warmup_steps:
        return max_lr * (i+1) / warmup_steps
    if i > max_steps:
        return min_lr
    decay_ratio = (i - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=DEVICE)


train_loader = DataLoaderLite(B=B, T=T, world_size=ddp_world_size, process_rank=ddp_local_rank, split="train")
val_loader = DataLoaderLite(B=B, T=T, world_size=ddp_world_size, process_rank=ddp_local_rank, split="val")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.csv")
with open(log_file, "w") as f: 
    f.write("step,train_loss,val_loss\n")


for i in range(max_steps):
    t0 = time.time()

    val_loss_accum = torch.tensor(torch.nan)

    if i % 250 == 0 or i == max_steps - 1:
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device=DEVICE, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            torch.distributed.all_reduce(val_loss_accum, op=torch.distributed.ReduceOp.AVG) # average val loss accross all processes
        if master_process:
            print(f"step {i} | val loss {val_loss_accum.item():.4f}")

    
    if (i % 1000 == 0 or i == max_steps - 1) and False:
        model.eval()
        num_return_sequences = 3
        max_length = 32
        enc = tiktoken.get_encoding("gpt2")
        x = torch.tensor(enc.encode("Hello, I'm a language model,"), dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(DEVICE)
        sample_gen = torch.Generator(device=DEVICE)
        sample_gen.manual_seed(42+ddp_rank)
        with torch.no_grad():
            while x.shape[1] < max_length:
                logits, _ = model(x)
                logits = logits[:, -1, :] # last token 
                probs = torch.nn.functional.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_gen)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat([x, xcol], dim=1)

        for j in range(num_return_sequences):
            tokens = x[j, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"iteration {i} | rank {ddp_rank} | sample {j} | {decoded}")
        model.train()


    if i % 5000 == 0 or i == max_steps - 1:
        checkpoint_path = os.path.join(log_dir, f"model_{i}.pt")
        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": raw_model.config,
            "step": i,
            "val_loss": val_loss_accum.item()
        }

    
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.
    for micro_step in range(grad_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(device=DEVICE, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accumulation_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accumulation_steps - 1
        loss.backward()
    if ddp:
        torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG) # average loss accross all processes
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip norm
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_per_sec = train_loader.B * train_loader.T * grad_accumulation_steps * ddp_world_size / dt
    if master_process:
        print(f"step {i:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{i},{loss_accum.item()},{val_loss_accum.item()}\n")

if ddp:
    torch.distributed.destroy_process_group()
