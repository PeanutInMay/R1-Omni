import os
import gradio as gr
import tempfile
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class EmotionAnalyzer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.bert_tokenizer = None
        self.model_loaded = False
        
    def load_model(self):
        if not self.model_loaded:
            # 初始化BERT分词器
            bert_model = "/home/zhzhu/model/bert-base-uncased"
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
            
            # 禁用Torch初始化
            disable_torch_init()
            
            # 初始化模型、处理器和分词器
            model_path = "/home/zhzhu/model/R1-Omni-0.5B"
            self.model, self.processor, self.tokenizer = model_init(model_path)
            self.model_loaded = True
    
    def analyze_emotion(self, video_file, modal_type):
        try:
            # 确保模型已加载
            if not self.model_loaded:
                self.load_model()
            
            # 使用默认指令
            instruct = "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
            
            # 处理视频输入
            video_tensor = self.processor['video'](video_file)
            
            # 根据modal类型决定是否处理音频
            audio_text = ""
            if modal_type == 'video_audio' or modal_type == 'audio':
                audio = self.processor['audio'](video_file)[0]
                
                # 获取音频转录文本
                if hasattr(self.model.get_audio_tower(), 'transcribe_audio'):
                    audio_text = self.model.get_audio_tower().transcribe_audio(audio)
            else:
                audio = None
            
            # 执行推理
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

# 创建分析器实例
analyzer = EmotionAnalyzer()

def analyze_video(video_file, modal_type):
    if video_file is None:
        return "请上传视频文件", ""
    
    emotion_result, audio_text = analyzer.analyze_emotion(video_file, modal_type)
    return emotion_result, audio_text

# 创建Gradio界面
with gr.Blocks(title="视频情感分析 Demo") as demo:
    gr.Markdown("# 🎭 视频情感分析系统")
    gr.Markdown("上传视频文件，系统将分析视频中人物的情感表达")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 视频上传组件
            video_input = gr.Video(
                label="上传视频文件",
                format="mp4"
            )
            
            # 模态选择
            modal_select = gr.Radio(
                choices=["video", "video_audio", "audio"],
                value="video_audio",
                label="分析模态",
                info="选择要分析的模态类型"
            )
            
            # 分析按钮
            analyze_btn = gr.Button("开始分析", variant="primary")
            
        with gr.Column(scale=1):
            # 音频转录结果显示
            audio_text_output = gr.Textbox(
                label="音频转录文本",
                lines=5,
                max_lines=8,
                placeholder="当选择包含音频的模态时，这里将显示音频转录的文本内容"
            )
            
            # 情感分析结果显示
            result_output = gr.Textbox(
                label="情感分析结果",
                lines=12,
                max_lines=15
            )
    
    # 示例区域
    gr.Markdown("## 📝 使用说明")
    gr.Markdown("""
    1. **上传视频**: 点击视频上传区域，选择要分析的视频文件
    2. **选择模态**: 
       - `video`: 仅分析视频画面
       - `video_audio`: 同时分析视频和音频（推荐）
       - `audio`: 仅分析音频
    3. **开始分析**: 点击分析按钮获取结果
    
    **注意**: 
    - 首次使用时模型加载可能需要一些时间，请耐心等待
    - 当选择包含音频的模态时，系统会自动转录音频内容并显示在上方文本框中
    """)
    
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, modal_select],
        outputs=[result_output, audio_text_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口号
        share=False,            # 设置为True可以生成公开链接
        debug=True              # 调试模式
    )