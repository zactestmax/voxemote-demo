from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os

from .models import get_model
from .preprocessing import load_audio_from_bytes
from .schemas import EmotionPrediction, HealthCheck

app = FastAPI(
    title="VoxEmote: Copilot Demo API",
    description="Демо API для распознавания эмоций по речи",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    get_model()

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Проверка работоспособности API"""
    model_instance = get_model()
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": model_instance.model.config._name_or_path
    }

@app.post("/predict_emotion/", response_model=EmotionPrediction)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Распознавание эмоции из аудиофайла
    
    - **file**: WAV файл (16 кГц, моно)
    - **return**: Эмоция, уверенность, все вероятности
    """
    # Проверка типа файла
    if not file.filename.endswith(('.wav', '.WAV')):
        raise HTTPException(status_code=400, detail="Только WAV файлы поддерживаются")
    
    try:
        # Чтение файла
        audio_bytes = await file.read()
        
        # Предобработка
        audio, sr = load_audio_from_bytes(audio_bytes)
        
        # Предсказание
        model_instance = get_model()
        result = model_instance.predict(audio, sr)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/")
async def root():
    """Информация о сервисе"""
    return {
        "service": "VoxEmote: Copilot Demo",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict_emotion/"
        }
    }