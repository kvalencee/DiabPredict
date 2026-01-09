"""
Servicio de predicción de diabetes usando modelos de Machine Learning
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class PredictionService:
    """Servicio para realizar predicciones de riesgo de diabetes"""

    def __init__(self, models_dir: Path):
        """
        Inicializa el servicio cargando los modelos entrenados

        Args:
            models_dir: Directorio donde están los modelos serializados
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self._load_models()

    def _load_models(self):
        """Carga los modelos de Machine Learning"""
        try:
            # Cargar escalador
            self.scaler = joblib.load(self.models_dir / 'scaler.pkl')

            # Cargar modelos
            self.models['logistic_regression'] = joblib.load(
                self.models_dir / 'logistic_regression.pkl'
            )
            self.models['random_forest'] = joblib.load(
                self.models_dir / 'random_forest.pkl'
            )
            self.models['svm'] = joblib.load(
                self.models_dir / 'svm.pkl'
            )

            print("[OK] Modelos de ML cargados exitosamente")

        except Exception as e:
            print(f"[ERROR] Error al cargar modelos: {e}")
            raise

    def predict(self, parameters: Dict[str, float]) -> Dict:
        """
        Realiza predicción de riesgo de diabetes

        Args:
            parameters: Diccionario con los parámetros clínicos

        Returns:
            Diccionario con el resultado de la predicción
        """
        # Preparar características en el orden correcto
        features = [
            parameters['pregnancies'],
            parameters['glucose'],
            parameters['blood_pressure'],
            parameters['skin_thickness'],
            parameters['insulin'],
            parameters['bmi'],
            parameters['pedigree_function'],
            parameters['age']
        ]

        # Convertir a array numpy y normalizar
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)

        # Obtener predicciones de cada modelo
        lr_prob = self.models['logistic_regression'].predict_proba(features_scaled)[0][1]
        rf_prob = self.models['random_forest'].predict_proba(features_scaled)[0][1]
        svm_prob = self.models['svm'].predict_proba(features_scaled)[0][1]

        # Calcular promedio (ensemble)
        avg_probability = (lr_prob + rf_prob + svm_prob) / 3.0

        # Clasificar riesgo
        risk_level = self._classify_risk(avg_probability)

        # Generar recomendaciones
        recommendations = self._generate_recommendations(parameters, risk_level)

        return {
            'risk_level': risk_level,
            'probability': round(avg_probability * 100, 2),
            'individual_predictions': {
                'logistic_regression': round(lr_prob * 100, 2),
                'random_forest': round(rf_prob * 100, 2),
                'svm': round(svm_prob * 100, 2)
            },
            'recommendations': recommendations
        }

    def _classify_risk(self, probability: float) -> str:
        """
        Clasifica el nivel de riesgo según la probabilidad

        Args:
            probability: Probabilidad de diabetes (0-1)

        Returns:
            Nivel de riesgo: 'Bajo', 'Medio', o 'Alto'
        """
        if probability < 0.30:
            return 'Bajo'
        elif probability < 0.70:
            return 'Medio'
        else:
            return 'Alto'

    def _generate_recommendations(self, parameters: Dict, risk_level: str) -> List[str]:
        """
        Genera recomendaciones personalizadas basadas en los parámetros

        Args:
            parameters: Parámetros clínicos del usuario
            risk_level: Nivel de riesgo determinado

        Returns:
            Lista de recomendaciones
        """
        recommendations = []

        # Recomendaciones basadas en nivel de riesgo general
        if risk_level == 'Alto':
            recommendations.append(
                "[!] IMPORTANTE: Su nivel de riesgo es ALTO. "
                "Consulte con un médico lo antes posible para una evaluación completa."
            )
        elif risk_level == 'Medio':
            recommendations.append(
                "[!] ATENCIÓN: Su nivel de riesgo es MEDIO. "
                "Se recomienda consultar con un profesional de la salud para evaluación adicional."
            )
        else:
            recommendations.append(
                "[OK] Su nivel de riesgo actual es BAJO. "
                "Mantenga hábitos de vida saludables y realice chequeos médicos regulares."
            )

        # Recomendaciones basadas en parámetros específicos

        # Glucosa
        if parameters['glucose'] > 140:
            recommendations.append(
                "[!] Glucosa elevada: Su nivel de glucosa está por encima del rango normal. "
                "Reduzca el consumo de azúcares y carbohidratos refinados."
            )
        elif parameters['glucose'] > 100:
            recommendations.append(
                "[*] Glucosa en prediabetes: Monitoree su nivel de glucosa regularmente "
                "y mantenga una dieta balanceada baja en azúcares."
            )

        # IMC
        if parameters['bmi'] > 30:
            recommendations.append(
                "[!] Obesidad: Su IMC indica obesidad. "
                "La pérdida de peso puede reducir significativamente su riesgo de diabetes. "
                "Consulte con un nutricionista."
            )
        elif parameters['bmi'] > 25:
            recommendations.append(
                "[*] Sobrepeso: Su IMC indica sobrepeso. "
                "Incremente la actividad física y mantenga una dieta balanceada para alcanzar un peso saludable."
            )
        elif parameters['bmi'] < 18.5:
            recommendations.append(
                "[*] Bajo peso: Su IMC es menor al rango saludable. "
                "Consulte con un profesional de la salud."
            )

        # Presión arterial
        if parameters['blood_pressure'] > 90:
            recommendations.append(
                "[!] Presión arterial elevada: Reduzca el consumo de sal, "
                "mantenga un peso saludable y realice ejercicio regular. "
                "Consulte con su médico."
            )
        elif parameters['blood_pressure'] > 80:
            recommendations.append(
                "[*] Presión arterial en límite: Monitoree su presión arterial regularmente "
                "y mantenga hábitos de vida saludables."
            )

        # Edad
        if parameters['age'] > 45:
            recommendations.append(
                "[i] Edad: A partir de los 45 años el riesgo de diabetes aumenta. "
                "Se recomienda realizar chequeos médicos anuales incluyendo pruebas de glucosa."
            )

        # Insulina
        if parameters['insulin'] > 200:
            recommendations.append(
                "[*] Nivel de insulina elevado: Puede indicar resistencia a la insulina. "
                "Consulte con un endocrinólogo para evaluación detallada."
            )

        # Historial familiar
        if parameters['pedigree_function'] > 1.0:
            recommendations.append(
                "[FAM] Historial familiar significativo: Su función de pedigree diabético es elevada. "
                "El factor genético es importante, extreme las medidas preventivas."
            )

        # Recomendaciones generales saludables
        recommendations.append(
            "[+] Actividad física: Realice al menos 150 minutos de ejercicio moderado por semana "
            "(caminar, nadar, ciclismo)."
        )

        recommendations.append(
            "[+] Alimentación saludable: Consuma abundantes vegetales, frutas, granos enteros, "
            "proteínas magras y grasas saludables. Limite azúcares y alimentos procesados."
        )

        recommendations.append(
            "[+] Hidratación: Beba suficiente agua durante el día (6-8 vasos). "
            "Evite bebidas azucaradas y alcohol en exceso."
        )

        recommendations.append(
            "[+] Descanso: Duerma 7-8 horas diarias. El sueño insuficiente aumenta el riesgo de diabetes."
        )

        recommendations.append(
            "[+] No fumar: Si fuma, busque ayuda para dejar el tabaco. "
            "Fumar aumenta el riesgo de diabetes y sus complicaciones."
        )

        return recommendations


class RecommendationEngine:
    """Motor para generar recomendaciones personalizadas (clase alternativa más detallada)"""

    @staticmethod
    def generate_comprehensive_report(parameters: Dict, prediction_result: Dict) -> Dict:
        """
        Genera un reporte comprehensivo con análisis detallado

        Args:
            parameters: Parámetros clínicos
            prediction_result: Resultado de la predicción

        Returns:
            Diccionario con reporte detallado
        """
        report = {
            'risk_assessment': {
                'level': prediction_result['risk_level'],
                'probability': prediction_result['probability'],
                'interpretation': ''
            },
            'parameter_analysis': {},
            'action_plan': [],
            'follow_up': []
        }

        # Interpretación del riesgo
        if prediction_result['risk_level'] == 'Alto':
            report['risk_assessment']['interpretation'] = (
                "Su evaluación indica un riesgo elevado de desarrollar diabetes tipo 2. "
                "Es fundamental que consulte con un profesional médico para realizar "
                "pruebas diagnósticas completas y discutir opciones de prevención o tratamiento."
            )
        elif prediction_result['risk_level'] == 'Medio':
            report['risk_assessment']['interpretation'] = (
                "Su evaluación indica un riesgo moderado. Aunque no es alarmante, "
                "es importante tomar medidas preventivas y realizar seguimiento médico regular."
            )
        else:
            report['risk_assessment']['interpretation'] = (
                "Su evaluación indica un riesgo bajo en este momento. "
                "Continúe con hábitos saludables y chequeos médicos periódicos."
            )

        # Análisis de parámetros individuales
        report['parameter_analysis'] = {
            'glucose': RecommendationEngine._analyze_glucose(parameters['glucose']),
            'bmi': RecommendationEngine._analyze_bmi(parameters['bmi']),
            'blood_pressure': RecommendationEngine._analyze_blood_pressure(parameters['blood_pressure']),
            'age': RecommendationEngine._analyze_age(parameters['age'])
        }

        return report

    @staticmethod
    def _analyze_glucose(glucose: float) -> Dict:
        """Analiza nivel de glucosa"""
        if glucose < 100:
            return {'status': 'normal', 'message': 'Nivel de glucosa normal'}
        elif glucose < 126:
            return {'status': 'prediabetes', 'message': 'Nivel de glucosa en rango de prediabetes'}
        else:
            return {'status': 'elevated', 'message': 'Nivel de glucosa elevado'}

    @staticmethod
    def _analyze_bmi(bmi: float) -> Dict:
        """Analiza IMC"""
        if bmi < 18.5:
            return {'status': 'underweight', 'message': 'Bajo peso'}
        elif bmi < 25:
            return {'status': 'normal', 'message': 'Peso normal'}
        elif bmi < 30:
            return {'status': 'overweight', 'message': 'Sobrepeso'}
        else:
            return {'status': 'obese', 'message': 'Obesidad'}

    @staticmethod
    def _analyze_blood_pressure(bp: float) -> Dict:
        """Analiza presión arterial diastólica"""
        if bp < 80:
            return {'status': 'normal', 'message': 'Presión arterial normal'}
        elif bp < 90:
            return {'status': 'elevated', 'message': 'Presión arterial elevada'}
        else:
            return {'status': 'high', 'message': 'Hipertensión'}

    @staticmethod
    def _analyze_age(age: int) -> Dict:
        """Analiza factor de edad"""
        if age < 45:
            return {'status': 'lower_risk', 'message': 'Grupo de edad de menor riesgo'}
        elif age < 65:
            return {'status': 'moderate_risk', 'message': 'Grupo de edad de riesgo moderado'}
        else:
            return {'status': 'higher_risk', 'message': 'Grupo de edad de mayor riesgo'}
