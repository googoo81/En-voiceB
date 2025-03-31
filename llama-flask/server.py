from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
from io import BytesIO
from urllib.request import urlopen

# Flask 앱 설정
app = Flask(__name__)
CORS(app)  # CORS 허용 (Next.js에서 접근 가능하도록)

# 모델과 프로세서 불러오기
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    user_audio_url = data.get("audio_url", "")  # 오디오 URL이 있는 경우

    if user_audio_url:
        # 오디오 처리
        audio_data = urlopen(user_audio_url).read()
        audio_input = librosa.load(BytesIO(audio_data), sr=processor.feature_extractor.sampling_rate)[0]
        
        # 텍스트와 오디오를 함께 처리
        text = processor.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio_url": user_audio_url}]}], add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio_input], return_tensors="pt", padding=True)
        
        # 모델 응답 생성
        generate_ids = model.generate(**inputs, max_length=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response_message = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    else:
        # 텍스트만 처리
        response_message = f"AI 응답: {user_message[::-1]}"  # 예시로 메시지 뒤집기 응답

    return jsonify({"response": response_message})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
