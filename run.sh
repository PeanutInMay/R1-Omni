#!/bin/bash

source ~/.zshrc
source ~/miniconda3/bin/activate
cd ~
cd /home/zhzhu/myproject/R1-Omni
conda activate r1-omni

python inference.py --modal video_audio \
  --model_path /home/zhzhu/model/R1-Omni-0.5B \
  --video_path /home/zhzhu/myproject/R1-Omni/test/video/video.mp4 \
  --instruct "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
