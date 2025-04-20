from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import inspect
import torch

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
    