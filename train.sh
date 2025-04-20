#!/usr/bin/env bash
export PYTHONPATH="$(pwd)"

NUM_GPU=$(python - << 'EOF'
import torch
print(torch.cuda.device_count())
EOF
)

if (( NUM_GPU > 1 )); then
  echo "==> Detected $NUM_GPU GPUs. Running with DDP"
  torchrun --standalone --nproc_per_node=$NUM_GPU src/train_gpt.py "$@"

else
  echo "==> $NUM_GPU GPUs found. Running single-process"
  python src/train_gpt.py "$@"
fi
