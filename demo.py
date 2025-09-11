import os
import gradio as gr
import tempfile
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer
from chat_module.get_response import get_openai_response

class EmotionAnalyzer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.bert_tokenizer = None
        self.model_loaded = False
        
    def load_model(self):
        if not self.model_loaded:
            bert_model = "/home/zhzhu/model/bert-base-uncased"
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
            disable_torch_init()
            model_path = "/home/zhzhu/model/R1-Omni-0.5B"
            self.model, self.processor, self.tokenizer = model_init(model_path)
            self.model_loaded = True
    
    def analyze_emotion(self, video_file, modal_type):
        try:
            if not self.model_loaded:
                self.load_model()
            
            instruct = "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
            
            video_tensor = self.processor['video'](video_file)
            
            audio_text = ""
            if modal_type == 'video_audio' or modal_type == 'audio':
                audio = self.processor['audio'](video_file)[0]
                if hasattr(self.model.get_audio_tower(), 'transcribe_audio'):
                    audio_text = self.model.get_audio_tower().transcribe_audio(audio)
            else:
                audio = None
            
            output = mm_infer(
                video_tensor, 
                instruct, 
                model=self.model, 
                tokenizer=self.tokenizer, 
                modal=modal_type, 
                question=instruct, 
                bert_tokeni=self.bert_tokenizer, 
                do_sample=False, 
                audio=audio
            )
            
            return output, audio_text
            
        except Exception as e:
            return f"分析过程中出现错误: {str(e)}", ""

analyzer = EmotionAnalyzer()

def analyze_video(video_file, modal_type):
    if video_file is None:
        return "请上传视频文件", ""
    
    emotion_result, audio_text = analyzer.analyze_emotion(video_file, modal_type)
    return emotion_result, audio_text

def chat_with_gpt(emotion_result, audio_text):
    if not emotion_result or emotion_result == "请上传视频文件":
        return "请先进行情感分析"
    
    if not audio_text.strip():
        return "没有检测到音频转录文本，请确保视频包含音频内容"
    
    try:
        response = get_openai_response(emotion_result, audio_text)
        return response
    except Exception as e:
        return f"对话生成失败: {str(e)}"

# 创建Gradio界面
with gr.Blocks(title="视频情感分析 Demo") as demo:
    gr.Markdown("# 🎭 视频情感分析与智能对话系统")
    gr.Markdown("上传视频文件，系统将分析视频中人物的情感表达，并可以基于分析结果进行对话")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="上传视频文件",
                format="mp4"
            )
            
            modal_select = gr.Radio(
                choices=["video", "video_audio", "audio"],
                value="video_audio",
                label="分析模态",
                info="选择要分析的模态类型"
            )
            
            analyze_btn = gr.Button("开始分析", variant="primary")
            
        with gr.Column(scale=1):
            audio_text_output = gr.Textbox(
                label="音频转录文本",
                lines=5,
                max_lines=8,
                placeholder="当选择包含音频的模态时，这里将显示音频转录的文本内容"
            )
            
            result_output = gr.Textbox(
                label="情感分析结果",
                lines=12,
                max_lines=15
            )
    
    # 智能对话功能
    gr.Markdown("## 💬 智能对话")
    with gr.Row():
        with gr.Column(scale=1):
            chat_btn = gr.Button("基于音频文本生成对话", variant="secondary")
        
        with gr.Column(scale=2):
            chat_output = gr.Textbox(
                label="AI回复",
                lines=8,
                max_lines=12,
                placeholder="AI将基于情感分析结果和音频转录文本给出相应回复"
            )
    
    gr.Markdown("## 📝 使用说明")
    gr.Markdown("""
    1. **上传视频**: 点击视频上传区域，选择要分析的视频文件
    2. **选择模态**: 
       - `video`: 仅分析视频画面
       - `video_audio`: 同时分析视频和音频（推荐）
       - `audio`: 仅分析音频
    3. **开始分析**: 点击分析按钮获取结果
    4. **智能对话**: 在情感分析完成后，点击对话按钮基于音频转录文本生成AI回复
    
    **注意**: 
    - 首次使用时模型加载可能需要一些时间，请耐心等待
    - 对话功能需要先完成情感分析且音频转录有内容
    """)
    
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, modal_select],
        outputs=[result_output, audio_text_output]
    )
    
    chat_btn.click(
        fn=chat_with_gpt,
        inputs=[result_output, audio_text_output],
        outputs=[chat_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )