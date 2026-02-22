from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2Processor
import torch
import time
import numpy as np

# ✅ Рабочая модель для демо: аудио-классификация эмоций (английский, но функционал покажет)
MODEL_NAME = "superb/wav2vec2-base-superb-er"

# Маппинг меток эмоций для этой модели (7 классов из SUPERB)
LABEL_MAP = {
    0: "neutral",
    1: "calm", 
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgusted"
}

# Группировка в 3 класса для ТЗ проекта
EMOTION_GROUPS = {
    "neutral": ["neutral", "calm", "happy"],
    "stress": ["sad", "disgusted"],
    "aggression": ["angry", "fearful"]
}

class EmotionModel:
    def __init__(self):
        print(f"Загрузка модели {MODEL_NAME}...")
        try:
            # ✅ Используем AutoModelForAudioClassification для аудио-задач
            self.model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Модель загружена на {self.device}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    def predict(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Предсказание эмоции для аудио
        """
        start_time = time.time()
        
        # Извлечение признаков через feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Перемещение на устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Обработка результатов
        probs_array = probabilities[0].cpu().numpy()
        
        # Группировка в 3 класса для ТЗ
        emotion_probs = {group: 0.0 for group in ["neutral", "stress", "aggression"]}
        for idx, prob in enumerate(probs_array):
            base_emotion = LABEL_MAP.get(idx, "neutral")
            for group, emotions in EMOTION_GROUPS.items():
                if base_emotion in emotions:
                    emotion_probs[group] += float(prob)
        
        # Определение доминирующего класса
        dominant_emotion = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[dominant_emotion]
        processing_time = (time.time() - start_time) * 1000  # мс
        
        return {
            "emotion": dominant_emotion,
            "confidence": round(confidence, 4),
            "all_probabilities": {k: round(v, 4) for k, v in emotion_probs.items()},
            "processing_time_ms": round(processing_time, 2)
        }

# Глобальный экземпляр модели
model = None

def get_model() -> EmotionModel:
    global model
    if model is None:
        model = EmotionModel()
    return model