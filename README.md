# Multimodal Emotion Recognition Demo

## 🛠️ Environment Setup
参考setup.sh和requirements.txt

## 📺  Demo

1. 修改run.sh中的相关路径为你的路径，并替换为你的的api, 并在`/chat_module/get_response.py`中修改为你的base_url
2. 运行run.sh
3. 如果出错可以直接执行`python demo.py`

##  📝 使用说明

1. **上传视频**: 点击视频上传区域，选择要分析的视频文件
2. **选择模态**: 
   - `video`: 仅分析视频画面
   - `video_audio`: 同时分析视频和音频（默认）
   - `audio`: 仅分析音频
3. **开始分析**: 点击分析按钮获取结果
4. **智能对话**: 在情感分析完成后，点击对话按钮基于音频转录文本生成AI回复

**注意**: 
- 首次使用时模型加载可能需要一些时间，请耐心等待
- 对话功能需要先完成情感分析且音频转录有内容
