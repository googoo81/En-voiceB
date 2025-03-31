from gtts import gTTS
import json
import os

# 기존 JSON 파일 불러오기
with open("fine-tune-dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 저장할 폴더 생성
audio_folder = "audio_data"
os.makedirs(audio_folder, exist_ok=True)

# 텍스트를 오디오로 변환
for i, data in enumerate(dataset):
    text = data["input"]
    audio_path = os.path.join("./", audio_folder, f"audio_{i}.mp3")
    
    # TTS 변환
    tts = gTTS(text, lang="en")
    tts.save(audio_path)
    
    # 데이터셋 업데이트 (오디오 파일 경로로 변경)
    data["input"] = audio_path

# 변환된 데이터셋 저장
with open("fine-tune-dataset-updated.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print("✅ TTS 변환 완료! fine-tune-dataset-updated.json 생성됨.")
