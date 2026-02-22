from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional


class EmotionPrediction(BaseModel):
    """Схема ответа API для предсказания эмоции"""
    
    # Конфигурация Pydantic v2: отключаем защиту namespace "model_*"
    model_config = ConfigDict(protected_namespaces=())
    
    emotion: str
    """Доминирующая эмоция: 'neutral', 'stress' или 'aggression'"""
    
    confidence: float
    """Уверенность модели в предсказании (0.0–1.0)"""
    
    all_probabilities: Dict[str, float]
    """Все вероятности по классам эмоций"""
    
    processing_time_ms: Optional[float] = None
    """Время обработки запроса в миллисекундах (опционально)"""


class HealthCheck(BaseModel):
    """Схема ответа для проверки работоспособности API"""
    
    # Конфигурация Pydantic v2: отключаем защиту namespace "model_*"
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    """Статус сервиса: 'healthy' или 'unhealthy'"""
    
    model_loaded: bool
    """Флаг загрузки ML-модели"""
    
    model_name: str
    """Название загруженной модели"""