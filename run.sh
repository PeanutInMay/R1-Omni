source ~/.zshrc
source ~/miniconda3/bin/activate
cd ~
cd /home/zhzhu/myproject/R1-Omni
conda activate r1-omni

export CUDA_VISIBLE_DEVICES=0
export API_KEY=0
export OPENAI_API_KEY=sk-xxx

python demo.py