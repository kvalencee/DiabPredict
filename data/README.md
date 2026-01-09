# Directorio de Datos - DiabPredict

## Estructura

```
data/
├── raw/                    # Datos crudos (dataset original)
│   └── diabetes.csv       # Dataset Pima Indians Diabetes
├── processed/             # Datos procesados
│   └── evaluaciones.json  # Historial de evaluaciones (generado automáticamente)
├── download_dataset.py    # Script para descargar dataset
└── README.md             # Este archivo
```

## Dataset: Pima Indians Diabetes

### Descripción
El dataset contiene información médica de mujeres de la tribu Pima Indians de al menos 21 años de edad. Se utiliza para predecir si una paciente tiene diabetes basándose en medidas diagnósticas.

### Características (8 variables predictoras)
1. **Pregnancies**: Número de embarazos
2. **Glucose**: Concentración de glucosa en plasma (mg/dL)
3. **BloodPressure**: Presión arterial diastólica (mmHg)
4. **SkinThickness**: Grosor del pliegue cutáneo del tríceps (mm)
5. **Insulin**: Insulina sérica de 2 horas (µU/mL)
6. **BMI**: Índice de masa corporal (kg/m²)
7. **DiabetesPedigreeFunction**: Función de pedigree de diabetes
8. **Age**: Edad (años)

### Variable Objetivo
- **Outcome**: 0 (no diabetes) o 1 (diabetes)

### Estadísticas
- **Instancias totales**: 768
- **Casos positivos**: 268 (34.9%)
- **Casos negativos**: 500 (65.1%)

## Descarga del Dataset

### Opción 1: Automática (Recomendada)
```bash
python data/download_dataset.py
```

### Opción 2: Manual
1. Visita: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Descarga el archivo `diabetes.csv`
3. Colócalo en `data/raw/diabetes.csv`

## Historial de Evaluaciones

El archivo `data/processed/evaluaciones.json` almacena todas las evaluaciones realizadas por los usuarios. Se genera automáticamente cuando se realiza la primera evaluación.

### Formato
```json
{
  "evaluations": [
    {
      "id": "uuid-string",
      "timestamp": "ISO-8601-datetime",
      "parameters": {
        "pregnancies": 6,
        "glucose": 148.0,
        ...
      },
      "result": {
        "risk_level": "Alto",
        "probability": 85.5,
        "recommendations": [...]
      }
    }
  ]
}
```

## Notas Importantes

⚠️ **Privacidad**: El historial de evaluaciones se almacena localmente. No se envía información a servidores externos.

⚠️ **Uso Médico**: Este dataset y las predicciones son solo para fines educativos y de investigación. NO deben usarse como diagnóstico médico real.

## Referencias

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
- Kaggle Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
