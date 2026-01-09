"""
Script para evaluar los modelos entrenados de DiabPredict
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import json

# Agregar el directorio raíz al path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config import Config

# Configuración
DATA_FILE = ROOT_DIR / 'data' / 'raw' / 'diabetes.csv'
MODELS_DIR = ROOT_DIR / 'ml' / 'models'


def load_models():
    """Carga todos los modelos entrenados"""
    print("=" * 70)
    print("CARGANDO MODELOS ENTRENADOS")
    print("=" * 70)
    
    models = {}
    model_files = {
        'Regresión Logística': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'SVM': 'svm.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = MODELS_DIR / filename
        if filepath.exists():
            models[name] = joblib.load(filepath)
            print(f"✓ {name} cargado")
        else:
            print(f"✗ {name} no encontrado en {filepath}")
    
    # Cargar scaler
    scaler_path = MODELS_DIR / 'scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler cargado")
    else:
        print(f"✗ Scaler no encontrado")
        scaler = None
    
    return models, scaler


def load_data():
    """Carga y prepara los datos de prueba"""
    print("\n" + "=" * 70)
    print("CARGANDO DATOS")
    print("=" * 70)
    
    # Cargar dataset
    df = pd.read_csv(DATA_FILE)
    print(f"✓ Dataset cargado: {len(df)} instancias")
    
    # Separar características y objetivo
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Usar los últimos 15% como conjunto de prueba (similar al entrenamiento)
    test_size = int(len(df) * 0.15)
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    
    print(f"✓ Conjunto de prueba: {len(X_test)} instancias")
    print(f"  - Casos positivos: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    print(f"  - Casos negativos: {len(y_test) - y_test.sum()} ({(len(y_test) - y_test.sum())/len(y_test)*100:.1f}%)")
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """Evalúa un modelo individual"""
    print("\n" + "=" * 70)
    print(f"EVALUANDO: {model_name.upper()}")
    print("=" * 70)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📊 Métricas de Rendimiento:")
    print(f"  Precisión (Accuracy):  {accuracy*100:6.2f}%")
    print(f"  Exactitud (Precision): {precision*100:6.2f}%")
    print(f"  Sensibilidad (Recall): {recall*100:6.2f}%")
    print(f"  F1-Score:              {f1*100:6.2f}%")
    
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"  AUC-ROC:               {auc*100:6.2f}%")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Matriz de Confusión:")
    print(f"  TN: {cm[0,0]:3d}  |  FP: {cm[0,1]:3d}")
    print(f"  FN: {cm[1,0]:3d}  |  TP: {cm[1,1]:3d}")
    
    # Reporte de clasificación
    print(f"\n📄 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc if y_pred_proba is not None else None,
        'confusion_matrix': cm.tolist()
    }


def compare_models(results):
    """Compara los resultados de todos los modelos"""
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 70)
    
    # Crear tabla comparativa
    print(f"\n{'Modelo':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']*100:>9.2f}% "
              f"{metrics['precision']*100:>9.2f}% "
              f"{metrics['recall']*100:>9.2f}% "
              f"{metrics['f1']*100:>9.2f}%")
    
    # Encontrar el mejor modelo
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print("\n" + "=" * 70)
    print(f"🏆 MEJOR MODELO: {best_model[0]}")
    print(f"   Precisión: {best_model[1]['accuracy']*100:.2f}%")
    print("=" * 70)


def main():
    """Función principal"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "DIABPREDICT - EVALUACIÓN DE MODELOS" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Cargar modelos
    models, scaler = load_models()
    
    if not models:
        print("\n❌ No se encontraron modelos entrenados.")
        print("💡 Ejecuta primero: python ml/scripts/train_models.py")
        return
    
    # Cargar datos
    X_test, y_test = load_data()
    
    # Escalar datos si hay scaler
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Evaluar cada modelo
    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(model, X_test_scaled, y_test, model_name)
    
    # Comparar modelos
    compare_models(results)
    
    # Guardar resultados
    results_file = MODELS_DIR / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        # Convertir numpy arrays a listas para JSON
        json_results = {}
        for name, metrics in results.items():
            json_results[name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'auc': float(metrics['auc']) if metrics['auc'] is not None else None,
                'confusion_matrix': metrics['confusion_matrix']
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Resultados guardados en: {results_file}")
    print("\n✅ Evaluación completada exitosamente\n")


if __name__ == '__main__':
    main()
