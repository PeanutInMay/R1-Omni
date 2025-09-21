source ~/.zshrc
source ~/miniconda3/bin/activate
cd ~
cd your_project_path
conda activate your_conda_env

export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=your_api_key

python demo.py