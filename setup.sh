pip3 install torch torchvision torchaudio
conda install conda-forge::transformers conda-forge::datasets conda-forge::tiktoken tqdm requests matplotlib -y
torchrun --standalone --nproc_per_node=8 train.py