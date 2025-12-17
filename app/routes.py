"""
Rutas de la aplicación DiabPredict
"""
from flask import Blueprint, render_template, request, jsonify, current_app
from app.services import PredictionService
from app.models import Evaluation, EvaluationManager
from datetime import datetime

# Crear Blueprint
main_bp = Blueprint('main', __name__)

# Instancias globales (se inicializan cuando se crea la app)
prediction_service = None
evaluation_manager = None


def init_services(app):
    """Inicializa los servicios globales"""
    global prediction_service, evaluation_manager

    prediction_service = PredictionService(app.config['ML_MODELS_DIR'])
    evaluation_manager = EvaluationManager(app.config['EVALUACIONES_FILE'])


@main_bp.record
def on_load(state):
    """Se ejecuta cuando se registra el blueprint"""
    init_services(state.app)


@main_bp.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@main_bp.route('/formulario')
def formulario():
    """Página del formulario de evaluación"""
    parameter_ranges = current_app.config['PARAMETER_RANGES']
    return render_template('formulario.html', ranges=parameter_ranges)


@main_bp.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicción

    Espera JSON con los siguientes campos:
    - pregnancies: int
    - glucose: float
    - blood_pressure: float
    - skin_thickness: float
    - insulin: float
    - bmi: float
    - pedigree_function: float
    - age: int
    """
    try:
        # Obtener datos del request
        data = request.get_json()

        # Validar que todos los campos estén presentes
        required_fields = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'pedigree_function', 'age'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Campo requerido faltante: {field}'
                }), 400

        # Validar rangos
        validation_errors = validate_parameters(data)
        if validation_errors:
            return jsonify({
                'error': 'Validación fallida',
                'details': validation_errors
            }), 400

        # Preparar parámetros
        parameters = {
            'pregnancies': float(data['pregnancies']),
            'glucose': float(data['glucose']),
            'blood_pressure': float(data['blood_pressure']),
            'skin_thickness': float(data['skin_thickness']),
            'insulin': float(data['insulin']),
            'bmi': float(data['bmi']),
            'pedigree_function': float(data['pedigree_function']),
            'age': float(data['age'])
        }

        # Realizar predicción
        result = prediction_service.predict(parameters)

        # Guardar evaluación
        evaluation = Evaluation(parameters=parameters, result=result)
        evaluation_manager.save_evaluation(evaluation)

        # Retornar resultado
        return jsonify({
            'success': True,
            'evaluation_id': evaluation.id,
            'result': result
        })

    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'details': str(e)
        }), 500


@main_bp.route('/resultados/<evaluation_id>')
def resultados(evaluation_id):
    """Página de resultados de una evaluación específica"""
    evaluation = evaluation_manager.get_evaluation_by_id(evaluation_id)

    if not evaluation:
        return render_template('error.html',
                               message='Evaluación no encontrada'), 404

    return render_template('resultados.html', evaluation=evaluation)


@main_bp.route('/historial')
def historial():
    """Página del historial de evaluaciones"""
    evaluations = evaluation_manager.get_all_evaluations()
    stats = evaluation_manager.get_statistics()

    return render_template('historial.html',
                           evaluations=evaluations,
                           stats=stats)


@main_bp.route('/api/evaluations')
def get_evaluations():
    """Endpoint para obtener todas las evaluaciones"""
    evaluations = evaluation_manager.get_all_evaluations()
    return jsonify({
        'evaluations': [e.to_dict() for e in evaluations]
    })


@main_bp.route('/api/evaluations/<evaluation_id>', methods=['GET'])
def get_evaluation(evaluation_id):
    """Endpoint para obtener una evaluación específica"""
    evaluation = evaluation_manager.get_evaluation_by_id(evaluation_id)

    if not evaluation:
        return jsonify({'error': 'Evaluación no encontrada'}), 404

    return jsonify(evaluation.to_dict())


@main_bp.route('/api/evaluations/<evaluation_id>', methods=['DELETE'])
def delete_evaluation(evaluation_id):
    """Endpoint para eliminar una evaluación"""
    success = evaluation_manager.delete_evaluation(evaluation_id)

    if success:
        return jsonify({'message': 'Evaluación eliminada exitosamente'})
    else:
        return jsonify({'error': 'No se pudo eliminar la evaluación'}), 500


@main_bp.route('/api/statistics')
def get_statistics():
    """Endpoint para obtener estadísticas"""
    stats = evaluation_manager.get_statistics()
    return jsonify(stats)


@main_bp.route('/ayuda')
def ayuda():
    """Página de ayuda"""
    parameter_ranges = current_app.config['PARAMETER_RANGES']
    return render_template('ayuda.html', ranges=parameter_ranges)


@main_bp.route('/info-modelos')
def info_modelos():
    """Página de información sobre los modelos"""
    model_info = {
        'logistic_regression': {
            'name': 'Regresión Logística',
            'description': 'Modelo estadístico que utiliza una función logística para clasificación binaria.',
            'strengths': [
                'Simple e interpretable',
                'Rápido de entrenar',
                'Proporciona probabilidades calibradas'
            ]
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensamble de múltiples árboles de decisión que combina sus predicciones.',
            'strengths': [
                'Robusto contra overfitting',
                'Maneja bien features no lineales',
                'Proporciona importancia de características'
            ]
        },
        'svm': {
            'name': 'Support Vector Machine',
            'description': 'Algoritmo que encuentra el hiperplano óptimo de separación entre clases.',
            'strengths': [
                'Efectivo en espacios de alta dimensión',
                'Robusto con datos no lineales (kernel RBF)',
                'Buena generalización'
            ]
        }
    }

    dataset_info = {
        'name': 'Pima Indians Diabetes Database',
        'source': 'UCI Machine Learning Repository',
        'instances': 768,
        'features': 8,
        'target': 'Outcome (0=No diabetes, 1=Diabetes)'
    }

    return render_template('info_modelos.html',
                           models=model_info,
                           dataset=dataset_info)


def validate_parameters(params: dict) -> list:
    """
    Valida los parámetros clínicos

    Args:
        params: Diccionario con los parámetros

    Returns:
        Lista de errores (vacía si no hay errores)
    """
    errors = []
    parameter_ranges = current_app.config['PARAMETER_RANGES']

    for param_name, param_value in params.items():
        if param_name not in parameter_ranges:
            continue

        range_config = parameter_ranges[param_name]

        try:
            # Convertir al tipo correcto
            if range_config['type'] == int:
                value = int(param_value)
            else:
                value = float(param_value)

            # Validar rango
            if value < range_config['min'] or value > range_config['max']:
                errors.append({
                    'field': param_name,
                    'message': f'Valor fuera de rango ({range_config["min"]}-{range_config["max"]})'
                })

        except (ValueError, TypeError):
            errors.append({
                'field': param_name,
                'message': 'Valor inválido'
            })

    return errors


@main_bp.app_errorhandler(404)
def not_found(error):
    """Manejador de errores 404"""
    return render_template('error.html',
                           message='Página no encontrada'), 404


@main_bp.app_errorhandler(500)
def internal_error(error):
    """Manejador de errores 500"""
    return render_template('error.html',
                           message='Error interno del servidor'), 500