import librosa
import numpy as np
from typing import Tuple

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SEC = 10

def load_and_preprocess_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Загрузка и предобработка аудиофайла
    """
    audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    
    # Ограничение длительности
    max_samples = TARGET_SAMPLE_RATE * MAX_DURATION_SEC
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    
    # Нормализация
    audio = librosa.util.normalize(audio)
    
    return audio, TARGET_SAMPLE_RATE

def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Загрузка аудио из байтов (для API)
    """
    import io
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SAMPLE_RATE, mono=True)
    return audio, TARGET_SAMPLE_RATE