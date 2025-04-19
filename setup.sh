pip install transformers datasets tiktoken tqdm matplotlib
sudo apt-get update
sudo apt-get install -y build-essential
torchrun --standalone --nproc_per_node=8 train_gpt2.py


mkdir src
cd src
git clone https://github.com/parthnayyar/GPT-2-from-scratch.git
cd GPT-2-from-scratch
pip install transformers datasets tiktoken tqdm matplotlib
sudo apt-get update
sudo apt-get install -y build-essential
clear
python fineweb.py