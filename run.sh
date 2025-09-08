#!/bin/bash

source ~/.bashrc
source ~/miniconda3/bin/activate
cd ~
cd /home/sxjiang/zhzhu/project/R1-Omni
conda activate r1-omni2

python inference.py --modal video_audio \
  --model_path /home/sxjiang/model/R1-Omni-0.5B \
  --video_path /home/sxjiang/zhzhu/project/R1-Omni/test/video/video.mp4 \
  --instruct "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
