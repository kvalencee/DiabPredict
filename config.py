"""
Archivo de configuración del proyecto DiabPredict
"""
import os
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Configuración base"""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False

    # Puerto del servidor
    PORT = int(os.environ.get('PORT', 5000))
    HOST = '127.0.0.1'

    # Directorios
    DATA_DIR = BASE_DIR / 'data'
    ML_MODELS_DIR = BASE_DIR / 'ml' / 'models'
    EVALUACIONES_FILE = DATA_DIR / 'processed' / 'evaluaciones.json'

    # Configuración de modelos ML
    MODEL_FILES = {
        'logistic_regression': ML_MODELS_DIR / 'logistic_regression.pkl',
        'random_forest': ML_MODELS_DIR / 'random_forest.pkl',
        'svm': ML_MODELS_DIR / 'svm.pkl',
        'scaler': ML_MODELS_DIR / 'scaler.pkl'
    }

    # Rangos de validación de parámetros clínicos
    PARAMETER_RANGES = {
        'pregnancies': {'min': 0, 'max': 20, 'type': int},
        'glucose': {'min': 0, 'max': 250, 'type': float},
        'blood_pressure': {'min': 40, 'max': 140, 'type': float},
        'skin_thickness': {'min': 0, 'max': 99, 'type': float},
        'insulin': {'min': 0, 'max': 850, 'type': float},
        'bmi': {'min': 10.0, 'max': 70.0, 'type': float},
        'pedigree_function': {'min': 0.078, 'max': 2.5, 'type': float},
        'age': {'min': 18, 'max': 120, 'type': int}
    }

    # Clasificación de riesgo
    RISK_THRESHOLDS = {
        'bajo': 0.30,
        'medio': 0.70
    }


class DevelopmentConfig(Config):
    """Configuración para desarrollo"""
    DEBUG = True
    TEMPLATES_AUTO_RELOAD = True


class ProductionConfig(Config):
    """Configuración para producción"""
    DEBUG = False


class TestingConfig(Config):
    """Configuración para pruebas"""
    TESTING = True
    DEBUG = True


# Diccionario de configuraciones
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}