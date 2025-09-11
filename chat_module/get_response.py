import os
from openai import OpenAI

def get_openai_response(emotion_result, user_text):
    client = OpenAI(
        base_url='https://api.oaipro.com/v1',
        api_key=os.getenv('OPENAI_API_KEY'),
    )
    
    prompt = f"""你需要借助用户的情感分析结果和用户输入的文本（ASR后的结果）综合分析，给出相应的回应。

情感分析结果：{emotion_result}

用户输入文本：{user_text}

请用中文回答，根据情感分析和用户输入给出合适的回应。"""
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
    )
    
    return response.choices[0].message.content