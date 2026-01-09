# 🏥 DiabPredict - Sistema de Predicción de Diabetes

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Sistema web de predicción de diabetes basado en Machine Learning utilizando el dataset Pima Indians Diabetes. Implementa tres modelos de clasificación (Regresión Logística, Random Forest y SVM) para evaluar el riesgo de diabetes en pacientes.

## 🌟 Características

- ✅ **Tres modelos de ML**: Regresión Logística, Random Forest y SVM
- 📊 **Interfaz web intuitiva**: Diseñada con Bootstrap 5
- 📈 **Visualización de resultados**: Gráficos y métricas detalladas
- 💾 **Historial de evaluaciones**: Almacenamiento local de predicciones
- 📱 **Diseño responsive**: Compatible con dispositivos móviles
- 🔒 **Privacidad**: Todos los datos se procesan localmente
- 📦 **Empaquetado**: Ejecutable standalone con PyInstaller

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Navegador web moderno

### Instalación

1. **Clonar o descargar el repositorio**
```bash
git clone <repository-url>
cd DiabPredict
```

2. **Crear entorno virtual**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar dataset**
```bash
python data/download_dataset.py
```

5. **Entrenar modelos**
```bash
python ml/scripts/train_models.py
```

6. **Ejecutar la aplicación**
```bash
python run.py
```

La aplicación se abrirá automáticamente en `http://127.0.0.1:5000`

## 📁 Estructura del Proyecto

```
DiabPredict/
│
├── app/                          # Aplicación Flask
│   ├── __init__.py              # Inicialización de Flask
│   ├── routes.py                # Rutas y controladores
│   ├── services.py              # Lógica de negocio
│   ├── models.py                # Modelos de datos
│   ├── static/                  # Archivos estáticos
│   │   ├── css/style.css       # Estilos personalizados
│   │   ├── js/main.js          # JavaScript
│   │   └── images/             # Imágenes
│   └── templates/               # Plantillas HTML
│       ├── base.html           # Plantilla base
│       ├── index.html          # Página principal
│       ├── formulario.html     # Formulario de evaluación
│       ├── resultados.html     # Resultados
│       ├── historial.html      # Historial
│       ├── ayuda.html          # Ayuda
│       └── info_modelos.html   # Información de modelos
│
├── data/                        # Datos
│   ├── raw/                    # Dataset original
│   │   └── diabetes.csv
│   ├── processed/              # Datos procesados
│   │   └── evaluaciones.json
│   ├── download_dataset.py     # Script de descarga
│   └── README.md
│
├── ml/                          # Machine Learning
│   ├── models/                 # Modelos entrenados (.pkl)
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   │   ├── svm.pkl
│   │   └── scaler.pkl
│   └── scripts/
│       ├── train_models.py     # Entrenamiento
│       └── evaluate_models.py  # Evaluación
│
├── tests/                       # Pruebas unitarias
│   ├── test_models.py
│   └── test_routes.py
│
├── config.py                    # Configuración
├── run.py                       # Punto de entrada
├── requirements.txt             # Dependencias
├── build.bat                    # Script de empaquetado
├── .gitignore                   # Archivos ignorados
└── README.md                    # Este archivo
```

## 🧪 Modelos de Machine Learning

### Dataset: Pima Indians Diabetes
- **Instancias**: 768
- **Características**: 8 (Embarazos, Glucosa, Presión Arterial, etc.)
- **Objetivo**: Predicción binaria (Diabetes / No Diabetes)

### Modelos Implementados

1. **Regresión Logística**
   - Modelo lineal simple y rápido
   - Precisión: ~77%
   - Mejor para interpretabilidad

2. **Random Forest**
   - Ensemble de árboles de decisión
   - Precisión: ~78%
   - Robusto ante overfitting

3. **SVM (Support Vector Machine)**
   - Clasificador basado en márgenes
   - Precisión: ~77%
   - Efectivo en espacios de alta dimensión

### Métricas de Evaluación
- Accuracy (Precisión)
- Precision (Exactitud)
- Recall (Sensibilidad)
- F1-Score
- AUC-ROC

## 📊 Uso de la Aplicación

### 1. Evaluación de Riesgo
1. Navega a "Nueva Evaluación"
2. Completa el formulario con los parámetros clínicos
3. Haz clic en "Evaluar Riesgo"
4. Revisa los resultados y recomendaciones

### 2. Historial
- Visualiza todas las evaluaciones anteriores
- Filtra por nivel de riesgo
- Exporta resultados a texto

### 3. Información de Modelos
- Consulta las métricas de rendimiento
- Compara los tres modelos
- Entiende cómo funcionan

## 🔧 Configuración

Edita `config.py` para personalizar:

```python
# Puerto del servidor
PORT = 5000

# Umbrales de riesgo
RISK_THRESHOLDS = {
    'bajo': 0.30,    # < 30% = Riesgo Bajo
    'medio': 0.70    # 30-70% = Medio, >70% = Alto
}
```

## 📦 Empaquetado

Para crear un ejecutable standalone:

```bash
# Windows
build.bat

# El ejecutable estará en dist/DiabPredict.exe
```

## 🧪 Pruebas

```bash
# Ejecutar todas las pruebas
python -m pytest tests/

# Con cobertura
python -m pytest tests/ --cov=app
```

## ⚠️ Advertencias Importantes

> **IMPORTANTE**: Este sistema es solo para fines educativos y de investigación. Las predicciones NO constituyen un diagnóstico médico y NO deben usarse como sustituto de la consulta con un profesional de la salud calificado.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👥 Autores

- **Tu Nombre** - *Desarrollo inicial*

## 🙏 Agradecimientos

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
- Framework: [Flask](https://flask.palletsprojects.com/)
- ML Library: [scikit-learn](https://scikit-learn.org/)
- UI Framework: [Bootstrap 5](https://getbootstrap.com/)

## 📞 Soporte

Si encuentras algún problema:

1. Revisa la [documentación completa](INSTALL.md)
2. Consulta la sección de [solución de problemas](INSTALL.md#solución-de-problemas)
3. Abre un [issue](https://github.com/tu-usuario/DiabPredict/issues)

---

**Desarrollado con ❤️ para la educación en Machine Learning y Salud Digital**