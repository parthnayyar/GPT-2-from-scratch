﻿# GPT-2-from-scratch

Reproduced GPT-2 124M from scratch in PyTorch with large-scale pre-training on the FineWeb EDU 10B dataset using Distributed Data Parallel (DDP).

## Introduction

This repository demonstrates a ground-up implementation of OpenAI’s GPT-2 (124M parameters) in PyTorch, inspired by [Andrej Karpathy’s YouTube walkthrough](https://www.youtube.com/watch?v=l8pRSuU81PU). You’ll find:

- A clean `src/` layout with `dataset`, `model`, and training scripts.
- End-to-end pre-training on a 10 billion-token academic web corpus.
- Performance optimizations: FlashAttention, `torch.compile`, `bfloat16` mixed precision, TF32 matmuls, power-of-2 vocab padding, fused Adam.
- Algorithmic optimizations: Gradient accumulation, LR warm-up & cosine decay, and norm clipping
- Improvements over the YouTube video: Dataset shuffling
- A modular README to guide you from setup to scaling to your own experiments.

## Features & Improvements

### Performance Optimizations

1. **TF32 MatMuls**  
   Enabled via `torch.set_float32_matmul_precision("high")` for ~3× faster matmul throughput.
2. **Mixed Precision**  
   Forward passes under `torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)` for reduced memory and further speedups.
3. **Compiled Graphs**  
   Wrapping the model with `torch.compile(model)` to fuse Python graph overhead.
4. **FlashAttention**  
   Custom attention kernel to maintain speed post-compilation.
5. **Power-of-2 Vocab Padding**  
   Padded vocabulary size from 50,257 → 50,304 to align CUDA kernels with block sizes.
6. **Fused Adam**  
   One-kernel weight updates via `fused=True` in optimizer config.

### Algorithmic Enhancements

- **Gradient Accumulation** to reach very large effective batch sizes without OOM.
- **Gradient Norm Clipping** (`clip_grad_norm_(…, 1.0)`) for stability.
- **Weight Decay & LR Schedule** following GPT-3’s warm-up + cosine decay.
- **Shard-based data loading** via `torch.utils.data.Dataset` + `DataLoader` with on-the-fly shuffle.

### Improvements over the YouTube video
- **Dataset shuffling** to promote unbiased training, ensure randomness in batch selection, and break inherent patterns

## Repository Structure

```
GPT-2-from-scratch/
├── assets/
│   └── loss.png                # Training/validation loss plot
├── data/
│   ├── train.bin                # Preprocessed FineWeb training tokens (generated after running fineweb.py)
│   ├── val.bin                  # Preprocessed FineWeb validation tokens (generated after running fineweb.py)
│   ├── shakespeare_train.npy    # Small Shakespeare train split (testing)
│   └── shakespeare_val.npy      # Small Shakespeare val split (testing)
├── logs/
│   ├── log.csv                  # Training logs (step, train_loss, val_loss)
│   ├── model_5000.pt            # Model checkpoint at 5k steps (generated after running src/train_gpt.py)
│   ├── model_10000.pt           # Model checkpoint at 10k steps (generated after running src/train_gpt.py)
│   ├── model_15000.pt           # Model checkpoint at 15k steps (generated after running src/train_gpt.py)
│   └── model_final.pt           # final Model checkpoint (generated after running src/train_gpt.py for 1 epoch)
├── src/
│   ├── __init__.py              # Marks src as a Python package
│   ├── dataset.py               # GPT2Dataset (torch.utils.data.Dataset)
│   ├── model.py                 # GPT2 model and GPTConfig
│   └── train_gpt.py             # Main training script
├── .gitignore                   # Ignores *.bin and *.pt files
├── fineweb.py                   # FineWeb dataset download/preprocessing
├── README.md                    # Project documentation (this file)
├── shakespeare.py               # Tiny dataset preparation script
├── shakespeare.txt              # Raw Shakespeare text
├── improvements.txt             # Additional notes/improvements log
└── train.sh                     # Bash script to launch training script
```

## Requirements

- Python 3.10+ (I used 3.12)
- PyTorch 2.x with CUDA support (I used 2.6)
- `tiktoken`, `transformers`, `datasets`, `tqdm`, `matplotlib`, `numpy`

## How to run training script

Rent some GPUs with PyTorch (cuDNN Runtime). Ensure torch is insured.

Once you are sshd into the system:
```bash
mkdir src # optionally make dir on system
cd src # change dir if you make the src dir
git clone https://github.com/parthnayyar/GPT-2-from-scratch.git # clone the repo
cd GPT-2-from-scratch
pip install transformers datasets tiktoken tqdm matplotlib # requirements
sudo apt-get update
sudo apt-get install -y build-essential # get c compiler to use torch.compile
clear
python fineweb.py # install fineweb dataset
chmod +x train.sh
./train.sh
```

## Training Details for my Training Run

- **Dataset:** FineWeb EDU 10B (10 billion GPT-2 tokens)
- **Compute:** 8× H100 SXM on [vast.ai](https://vast.ai)
- **Duration:** ~10 min to download & preprocess, ~1 hr for 20,000 steps (1 epoch is ~19,000 steps)
- **Cost:** ~$20 USD (download + training), ~$50 USD total including experiment setup.

## Results

### Training Loss Curve

The following plot shows the training loss and validation loss during 1 epoch of pretraining on the FineWebEdu-10B dataset.

![Training and Validation Loss](assets/loss.png)

---

### Sample Generations

Below are some random generations (truncated to first 32 generation tokens) from the model after training for 20,000 steps.

#### Prompt: "The secrets of the ancient temple were hidden behind"

> The secrets of the ancient temple were hidden behind the columns of rock formations. The priests, priests and the Levites were not allowed to come to the sacred temple. The people, fearing the coming of the

#### Prompt: "Artificial intelligence has the potential to"

> Artificial intelligence has the potential to enable computer-controlled robotic surgery to treat diseases which have not been successfully performed. Many advanced robotic systems including biometric machines, prosthetic limbs used for robotic assisted

#### Prompt: "I am a language model and I want to"

> I am a language model and I want to teach my students the structures of sound within that environment. I think we can teach our students to think with some clarity and to think with some precision as we move

---

Note: These generations are sampled after relatively short training (1 epoch). With more training and tuning, the model quality would continue to improve.

## What I Learned

- **Distributed Data Parallel (DDP):** Bootstrapping multi-GPU training, synchronizing gradients & metrics, efficient rank-based data loading.
- **Mixed-Precision & TF32:** Balancing precision/performance trade-offs with `autocast` and `torch.set_float32_matmul_precision`.
- **PyTorch 2.0 Compiler:** How `torch.compile` can drastically cut operator overhead in a deep transformer.
- **FlashAttention Integration:** Modifying attention internals so that compilation doesn’t degrade kernel performance.
- **Batching & Accumulation:** Computing effective batch size of 524,288 tokens via gradient accumulation without OOM.
- **LR Scheduling:** Implementing a combined linear warm-up + cosine decay per GPT-3 guidelines.
- **Data Pipeline:** Building a custom `Dataset`/`DataLoader` for huge token corpora with distributed computing, shuffling, and split semantics.

## Potential Future Improvements

- Increase `max_lr` by 2–3× to explore more aggressive convergence regimes.
- Scale context length beyond 1,024 tokens for longer memory.
- Integrate longer-sequence sparse attention for ultra-long documents.
- Benchmark with AdamW vs. newer optimizers like Lion or Adafactor.
- Add RoPE embeddings.
- Add KV cache for faster inference.
- Hyperparameter tuning via tools like Hydra or Optuna.
- Finetuning for chatbot capabilities by adding new special tokens.

## References

- [Language Models are Unsupervised Multitask Learners (GPT-2 Paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners (GPT-3 Paper)](https://arxiv.org/abs/2005.14165)
- [FineWebEdu-10B Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Attention Is All You Need (Transformer Paper)](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy's Video Tutorial on GPT](https://www.youtube.com/watch?v=l8pRSuU81PU)

## Acknowledgments

This implementation is inspired by Andrej Karpathy’s tutorial and his approach to making complex AI concepts more accessible.
