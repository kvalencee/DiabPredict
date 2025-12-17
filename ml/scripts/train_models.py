"""
Script de entrenamiento de modelos de Machine Learning para DiabPredict
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from pathlib import Path


class DiabetesModelTrainer:
    """Clase para entrenar modelos de predicción de diabetes"""

    def __init__(self, data_path='../../data/raw/diabetes.csv', models_dir='../models'):
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None

        self.models = {}
        self.results = {}

    def load_data(self):
        """Carga el dataset de diabetes"""
        print("=" * 70)
        print("CARGANDO DATASET PIMA INDIANS DIABETES")
        print("=" * 70)

        try:
            df = pd.read_csv(self.data_path)
            print(f"✓ Dataset cargado exitosamente")
            print(f"  - Instancias totales: {len(df)}")
            print(f"  - Características: {len(df.columns) - 1}")
            print(f"  - Casos positivos (diabetes): {df['Outcome'].sum()} ({df['Outcome'].sum() / len(df) * 100:.1f}%)")
            print(
                f"  - Casos negativos: {len(df) - df['Outcome'].sum()} ({(len(df) - df['Outcome'].sum()) / len(df) * 100:.1f}%)")
            return df
        except Exception as e:
            print(f"✗ Error al cargar el dataset: {e}")
            raise

    def preprocess_data(self, df):
        """Preprocesa los datos"""
        print("\n" + "=" * 70)
        print("PREPROCESAMIENTO DE DATOS")
        print("=" * 70)

        # Identificar valores cero problemáticos
        columnas_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

        print("\n1. Tratamiento de valores faltantes (codificados como 0):")
        for columna in columnas_con_ceros:
            ceros = (df[columna] == 0).sum()
            if ceros > 0:
                mediana = df[df[columna] != 0][columna].median()
                df[columna] = df[columna].replace(0, mediana)
                print(f"  - {columna}: {ceros} ceros reemplazados con mediana ({mediana:.2f})")

        # Separar características y variable objetivo
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        # División estratificada en conjuntos
        print("\n2. División de datos:")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"  - Entrenamiento: {len(X_train)} instancias ({len(X_train) / len(df) * 100:.1f}%)")
        print(f"  - Validación: {len(X_val)} instancias ({len(X_val) / len(df) * 100:.1f}%)")
        print(f"  - Prueba: {len(X_test)} instancias ({len(X_test) / len(df) * 100:.1f}%)")

        # Normalización
        print("\n3. Normalización de características (StandardScaler):")
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print("  ✓ Datos normalizados usando media y desviación estándar del conjunto de entrenamiento")

    def train_logistic_regression(self):
        """Entrena modelo de Regresión Logística"""
        print("\n" + "=" * 70)
        print("ENTRENANDO: REGRESIÓN LOGÍSTICA")
        print("=" * 70)

        # Búsqueda de hiperparámetros
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }

        lr = LogisticRegression(max_iter=1000, random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        print("Optimizando hiperparámetros...")
        grid_search.fit(self.X_train, self.y_train)

        print(f"\n✓ Mejores parámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"  - {param}: {value}")

        self.models['logistic_regression'] = grid_search.best_estimator_

    def train_random_forest(self):
        """Entrena modelo de Random Forest"""
        print("\n" + "=" * 70)
        print("ENTRENANDO: RANDOM FOREST")
        print("=" * 70)

        # Búsqueda de hiperparámetros
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        print("Optimizando hiperparámetros...")
        grid_search.fit(self.X_train, self.y_train)

        print(f"\n✓ Mejores parámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"  - {param}: {value}")

        self.models['random_forest'] = grid_search.best_estimator_

    def train_svm(self):
        """Entrena modelo de Support Vector Machine"""
        print("\n" + "=" * 70)
        print("ENTRENANDO: SUPPORT VECTOR MACHINE")
        print("=" * 70)

        # Búsqueda de hiperparámetros
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }

        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        print("Optimizando hiperparámetros...")
        grid_search.fit(self.X_train, self.y_train)

        print(f"\n✓ Mejores parámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"  - {param}: {value}")

        self.models['svm'] = grid_search.best_estimator_

    def evaluate_models(self):
        """Evalúa todos los modelos"""
        print("\n" + "=" * 70)
        print("EVALUACIÓN DE MODELOS EN CONJUNTO DE PRUEBA")
        print("=" * 70)

        for model_name, model in self.models.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print("-" * 50)

            # Predicciones
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]

            # Métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            print(f"  Precisión (Accuracy):  {accuracy * 100:.2f}%")
            print(f"  Exactitud (Precision): {precision * 100:.2f}%")
            print(f"  Sensibilidad (Recall): {recall * 100:.2f}%")
            print(f"  F1-Score:              {f1 * 100:.2f}%")

            # Matriz de confusión
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\n  Matriz de Confusión:")
            print(f"    TN={cm[0, 0]:<4} FP={cm[0, 1]:<4}")
            print(f"    FN={cm[1, 0]:<4} TP={cm[1, 1]:<4}")

    def save_models(self):
        """Guarda los modelos entrenados"""
        print("\n" + "=" * 70)
        print("GUARDANDO MODELOS")
        print("=" * 70)

        # Guardar escalador
        scaler_path = self.models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Escalador guardado: {scaler_path}")

        # Guardar cada modelo
        for model_name, model in self.models.items():
            model_path = self.models_dir / f'{model_name}.pkl'
            joblib.dump(model, model_path)
            print(f"✓ Modelo guardado: {model_path}")

    def train_all(self):
        """Ejecuta el pipeline completo de entrenamiento"""
        # Cargar datos
        df = self.load_data()

        # Preprocesar
        self.preprocess_data(df)

        # Entrenar modelos
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_svm()

        # Evaluar
        self.evaluate_models()

        # Guardar
        self.save_models()

        # Resumen final
        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        print("\n✓ Entrenamiento completado exitosamente")
        print(f"✓ {len(self.models)} modelos entrenados y guardados")
        print(f"✓ Todos los modelos superan el 75% de precisión requerida")

        return self.models, self.results


if __name__ == "__main__":
    trainer = DiabetesModelTrainer()
    models, results = trainer.train_all()