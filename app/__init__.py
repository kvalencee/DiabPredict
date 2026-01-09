"""
Inicialización de la aplicación Flask para DiabPredict
"""
from flask import Flask
from config import config


def create_app(config_name='default'):
    """
    Factory function para crear la aplicación Flask
    
    Args:
        config_name: Nombre de la configuración a usar ('development', 'production', 'testing')
    
    Returns:
        Flask app instance
    """
    # Crear instancia de Flask
    app = Flask(__name__)
    
    # Cargar configuración
    app.config.from_object(config[config_name])
    
    # Registrar blueprints/rutas
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
