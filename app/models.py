"""
Modelos de datos para DiabPredict
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import uuid


class Evaluation:
    """Modelo para una evaluación de riesgo de diabetes"""

    def __init__(self, parameters: Dict, result: Dict, evaluation_id: str = None):
        """
        Inicializa una evaluación

        Args:
            parameters: Parámetros clínicos ingresados
            result: Resultado de la predicción
            evaluation_id: ID único (se genera automáticamente si no se proporciona)
        """
        self.id = evaluation_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.timestamp = datetime.now().isoformat()
        self.parameters = parameters
        self.result = result

    def to_dict(self) -> Dict:
        """Convierte la evaluación a diccionario"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'result': self.result
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Evaluation':
        """Crea una evaluación desde un diccionario"""
        eval_obj = cls(
            parameters=data['parameters'],
            result=data['result'],
            evaluation_id=data['id']
        )
        eval_obj.timestamp = data['timestamp']
        return eval_obj


class EvaluationManager:
    """Gestor para almacenar y recuperar evaluaciones"""

    def __init__(self, file_path: Path):
        """
        Inicializa el gestor

        Args:
            file_path: Ruta al archivo JSON de almacenamiento
        """
        self.file_path = Path(file_path)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Asegura que el archivo de datos existe"""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_evaluations([])

    def _load_evaluations(self) -> List[Dict]:
        """Carga evaluaciones desde el archivo"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('evaluaciones', [])
        except json.JSONDecodeError:
            return []
        except Exception as e:
            print(f"Error al cargar evaluaciones: {e}")
            return []

    def _save_evaluations(self, evaluations: List[Dict]):
        """Guarda evaluaciones al archivo"""
        try:
            data = {'evaluaciones': evaluations}
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error al guardar evaluaciones: {e}")
            raise

    def save_evaluation(self, evaluation: Evaluation) -> bool:
        """
        Guarda una nueva evaluación

        Args:
            evaluation: Objeto Evaluation a guardar

        Returns:
            True si se guardó exitosamente, False en caso contrario
        """
        try:
            evaluations = self._load_evaluations()
            evaluations.append(evaluation.to_dict())
            self._save_evaluations(evaluations)
            return True
        except Exception as e:
            print(f"Error al guardar evaluación: {e}")
            return False

    def get_all_evaluations(self, sort_by='timestamp', ascending=False) -> List[Evaluation]:
        """
        Obtiene todas las evaluaciones

        Args:
            sort_by: Campo por el cual ordenar ('timestamp', 'risk_level')
            ascending: True para orden ascendente, False para descendente

        Returns:
            Lista de objetos Evaluation
        """
        evaluations_data = self._load_evaluations()
        evaluations = [Evaluation.from_dict(e) for e in evaluations_data]

        if sort_by == 'timestamp':
            evaluations.sort(key=lambda x: x.timestamp, reverse=not ascending)
        elif sort_by == 'risk_level':
            risk_order = {'Alto': 3, 'Medio': 2, 'Bajo': 1}
            evaluations.sort(
                key=lambda x: risk_order.get(x.result['risk_level'], 0),
                reverse=not ascending
            )

        return evaluations

    def get_evaluation_by_id(self, evaluation_id: str) -> Optional[Evaluation]:
        """
        Obtiene una evaluación específica por ID

        Args:
            evaluation_id: ID de la evaluación

        Returns:
            Objeto Evaluation o None si no se encuentra
        """
        evaluations = self._load_evaluations()
        for eval_data in evaluations:
            if eval_data['id'] == evaluation_id:
                return Evaluation.from_dict(eval_data)
        return None

    def delete_evaluation(self, evaluation_id: str) -> bool:
        """
        Elimina una evaluación por ID

        Args:
            evaluation_id: ID de la evaluación a eliminar

        Returns:
            True si se eliminó exitosamente, False en caso contrario
        """
        try:
            evaluations = self._load_evaluations()
            evaluations = [e for e in evaluations if e['id'] != evaluation_id]
            self._save_evaluations(evaluations)
            return True
        except Exception as e:
            print(f"Error al eliminar evaluación: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas de las evaluaciones

        Returns:
            Diccionario con estadísticas
        """
        evaluations = self.get_all_evaluations()

        if not evaluations:
            return {
                'total': 0,
                'by_risk_level': {'Alto': 0, 'Medio': 0, 'Bajo': 0},
                'average_probability': 0
            }

        stats = {
            'total': len(evaluations),
            'by_risk_level': {'Alto': 0, 'Medio': 0, 'Bajo': 0},
            'average_probability': 0
        }

        total_prob = 0
        for eval in evaluations:
            risk_level = eval.result['risk_level']
            stats['by_risk_level'][risk_level] += 1
            total_prob += eval.result['probability']

        stats['average_probability'] = round(total_prob / len(evaluations), 2)

        return stats

    def clear_all_evaluations(self) -> bool:
        """
        Elimina todas las evaluaciones

        Returns:
            True si se eliminaron exitosamente
        """
        try:
            self._save_evaluations([])
            return True
        except Exception as e:
            print(f"Error al limpiar evaluaciones: {e}")
            return False