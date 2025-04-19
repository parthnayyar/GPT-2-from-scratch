pip install torch torchvision torchaudio transformers datasets tiktoken tqdm matplotlib
sudo apt-get update
sudo apt-get install -y build-essential
torchrun --standalone --nproc_per_node=8 train.py