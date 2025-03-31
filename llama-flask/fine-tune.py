from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
import torchaudio
import librosa
import numpy as np

torchaudio.set_audio_backend("sox_io")

# 새로운 데이터셋 불러오기
dataset = Dataset.from_json("./fine-tune-dataset-updated.json")

# 모델 및 프로세서 로드
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

# 오디오 전처리 함수
def preprocess_function(examples):
    # 오디오 로드 및 리샘플링
    waveform, sample_rate = librosa.load(examples["input"], sr=processor.feature_extractor.sampling_rate)
    
    # 오디오 길이를 30초로 제한 (필요에 따라 조정 가능)
    max_length = 30 * processor.feature_extractor.sampling_rate
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    else:
        # 패딩
        padding_length = max_length - len(waveform)
        waveform = np.pad(waveform, (0, padding_length), mode='constant')
    
    # ChatML 형식으로 대화 구성
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "audio", "audio": waveform},
            {"type": "text", "text": "What is in the audio?"}
        ]},
        {"role": "assistant", "content": examples["output"]}
    ]
    
    # 텍스트 템플릿 적용
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # 오디오 특성 추출
    audio_features = processor.feature_extractor(
        waveform,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    
    # 텍스트 입력 처리
    text_inputs = processor.tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # 입력 데이터 구성
    inputs = {
        'input_ids': text_inputs['input_ids'].squeeze().tolist(),
        'attention_mask': text_inputs['attention_mask'].squeeze().tolist(),
        'input_features': audio_features['input_features'].squeeze().tolist(),
        'feature_attention_mask': audio_features['attention_mask'].squeeze().tolist()
    }
    
    return inputs

# 전처리 적용
dataset = dataset.map(preprocess_function, remove_columns=["input"])

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=processor.tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 모델 훈련
trainer.train()
