# 👤 Manual de Usuario - DiabPredict

## 📖 Guía Completa de Uso

### 🏠 Página Principal

Al abrir DiabPredict verás:
- **Descripción del sistema**: Qué hace DiabPredict
- **Cómo funciona**: Proceso de evaluación en 4 pasos
- **Beneficios**: Importancia de la detección temprana
- **Advertencia**: Recordatorio de que NO es diagnóstico médico

**Botones principales:**
- `Comenzar Evaluación` - Inicia una nueva evaluación
- `Cómo Funciona` - Abre la página de ayuda

---

## 📝 Realizar una Evaluación

### Paso 1: Acceder al Formulario
1. Haz clic en `Comenzar Evaluación` o en `Formulario` en el menú
2. Se abrirá el formulario con 8 parámetros clínicos

### Paso 2: Completar los Parámetros

#### 1️⃣ Número de Embarazos
- **Rango:** 0 - 20
- **Ejemplo:** 2
- **Nota:** Solo aplica a mujeres

#### 2️⃣ Glucosa en Plasma (mg/dL)
- **Rango:** 0 - 250
- **Ejemplo:** 120
- **Referencia:**
  - Normal: < 140
  - Prediabetes: 140-199
  - Diabetes: ≥ 200

#### 3️⃣ Presión Arterial Diastólica (mmHg)
- **Rango:** 40 - 140
- **Ejemplo:** 70
- **Referencia:**
  - Normal: < 80
  - Elevada: 80-89
  - Hipertensión: ≥ 90

#### 4️⃣ Grosor Pliegue Cutáneo (mm)
- **Rango:** 0 - 99
- **Ejemplo:** 20
- **Nota:** Medida del tríceps

#### 5️⃣ Insulina Sérica (µU/mL)
- **Rango:** 0 - 850
- **Ejemplo:** 80
- **Nota:** Medida 2 horas post-carga

#### 6️⃣ Índice de Masa Corporal (IMC)
- **Rango:** 10.0 - 70.0
- **Ejemplo:** 25.5
- **Cálculo:** Peso (kg) / Altura² (m²)
- **Referencia:**
  - Bajo peso: < 18.5
  - Normal: 18.5-24.9
  - Sobrepeso: 25-29.9
  - Obesidad: ≥ 30

#### 7️⃣ Función de Pedigree Diabético
- **Rango:** 0.078 - 2.5
- **Ejemplo:** 0.5
- **Nota:** Historial familiar de diabetes
  - Bajo: < 0.5
  - Medio: 0.5-1.0
  - Alto: > 1.0

#### 8️⃣ Edad (años)
- **Rango:** 18 - 120
- **Ejemplo:** 35

### Paso 3: Enviar el Formulario
1. Revisa que todos los campos estén completos
2. Haz clic en `Evaluar Riesgo`
3. Espera unos segundos mientras se procesa

---

## 📊 Interpretar Resultados

### Nivel de Riesgo

La evaluación te mostrará uno de tres niveles:

#### 🟢 Riesgo Bajo (< 30%)
- **Significado:** Probabilidad baja de diabetes
- **Acción:** Mantén hábitos saludables
- **Seguimiento:** Chequeos médicos regulares

#### 🟡 Riesgo Medio (30% - 70%)
- **Significado:** Riesgo moderado
- **Acción:** Consulta con un médico
- **Seguimiento:** Considera cambios en estilo de vida

#### 🔴 Riesgo Alto (> 70%)
- **Significado:** Riesgo elevado
- **Acción:** Consulta médica URGENTE
- **Seguimiento:** Evaluación profesional completa

### Predicciones por Modelo

Verás tres predicciones individuales:
- **Regresión Logística:** Modelo estadístico lineal
- **Random Forest:** Ensemble de árboles de decisión
- **SVM:** Máquina de vectores de soporte

**Probabilidad Final:** Promedio de los tres modelos

### Recomendaciones Personalizadas

El sistema genera recomendaciones basadas en:
- Tu nivel de riesgo general
- Valores específicos de tus parámetros
- Factores de riesgo identificados

**Tipos de recomendaciones:**
- 🏥 Consultas médicas
- 💪 Actividad física
- 🥗 Alimentación
- 💧 Hidratación
- 😴 Descanso
- 🚭 Hábitos saludables

---

## 📜 Ver Historial

### Acceder al Historial
1. Haz clic en `Historial` en el menú
2. Verás todas tus evaluaciones anteriores

### Estadísticas Generales
El historial muestra:
- **Total de evaluaciones**
- **Evaluaciones por nivel de riesgo:**
  - Riesgo Bajo (verde)
  - Riesgo Medio (amarillo)
  - Riesgo Alto (rojo)

### Ver Detalles de una Evaluación
1. Encuentra la evaluación en la lista
2. Haz clic en `Ver Detalles`
3. Se abrirá la página completa de resultados

### Información Mostrada
- Fecha y hora de la evaluación
- Nivel de riesgo y probabilidad
- Todos los parámetros ingresados
- Recomendaciones generadas

---

## ℹ️ Información de Modelos

### Acceder
Haz clic en `Prevención` en el menú

### Contenido
- **Dataset utilizado:** Pima Indians Diabetes
- **Modelos implementados:** Descripción de cada uno
- **Método Ensemble:** Cómo se combinan
- **Métricas de rendimiento:** Precisión de cada modelo
- **Detalles técnicos:** Tecnologías utilizadas
- **Limitaciones:** Qué debes saber

---

## ❓ Ayuda

### Acceder
Haz clic en `Ayuda` en el menú o en `Cómo Funciona`

### Contenido
- **Cómo funciona DiabPredict**
- **Explicación de cada parámetro clínico**
- **Interpretación de resultados**
- **Preguntas frecuentes**

---

## 🖨️ Imprimir Resultados

### Desde la Página de Resultados
1. Haz clic en el botón `Imprimir`
2. Se abrirá el diálogo de impresión
3. Selecciona tu impresora o "Guardar como PDF"
4. Confirma la impresión

**Nota:** La versión impresa omite elementos de navegación para mejor legibilidad

---

## ⚠️ Advertencias Importantes

### NO es un Diagnóstico Médico
- DiabPredict es una **herramienta educativa**
- Los resultados son **estimaciones estadísticas**
- **NO reemplaza** la consulta médica
- **Siempre consulta** con un profesional de salud

### Precisión del Sistema
- Los modelos tienen ~75% de precisión
- Aproximadamente 1 de cada 4 predicciones puede ser incorrecta
- Entrenado con población específica (Pima Indians)
- Puede tener menor precisión en otras poblaciones

### Privacidad
- Todos los datos se procesan **localmente**
- **No se envía información** a servidores externos
- El historial se guarda en tu dispositivo
- **No se comparte** con terceros

---

## 💡 Consejos de Uso

### Para Mejores Resultados
1. ✅ Usa valores de análisis médicos recientes
2. ✅ Ingresa datos precisos y verificados
3. ✅ Consulta con tu médico sobre los valores
4. ✅ Realiza evaluaciones periódicas para seguimiento
5. ❌ No inventes valores si no los conoces

### Cuándo Realizar una Evaluación
- Después de análisis médicos
- Como seguimiento de cambios en estilo de vida
- Antes de consulta médica (para discutir resultados)
- Periódicamente si tienes factores de riesgo

### Qué Hacer con los Resultados
1. **Guárdalos o imprímelos** para tu registro
2. **Compártelos con tu médico** en consulta
3. **Úsalos como motivación** para hábitos saludables
4. **NO los uses** como diagnóstico definitivo

---

## 🔄 Flujo de Trabajo Recomendado

```
1. Obtener análisis médicos
   ↓
2. Ingresar datos en DiabPredict
   ↓
3. Revisar resultados y recomendaciones
   ↓
4. Imprimir o guardar resultados
   ↓
5. Consultar con médico
   ↓
6. Seguir recomendaciones médicas
   ↓
7. Realizar seguimiento periódico
```

---

## 📞 Preguntas Frecuentes

### ¿Con qué frecuencia debo hacer una evaluación?
Depende de tu situación. Consulta con tu médico.

### ¿Puedo confiar en los resultados?
Son estimaciones con ~75% de precisión. Siempre consulta con un médico.

### ¿Mis datos están seguros?
Sí, todo se procesa localmente en tu dispositivo.

### ¿Qué hago si obtengo riesgo alto?
Consulta con un médico lo antes posible. No te alarmes, pero no ignores el resultado.

### ¿Puedo usar esto para diagnosticar diabetes?
NO. Solo un médico puede diagnosticar diabetes.

---

## 🎯 Resumen Rápido

1. **Abrir aplicación:** `python run.py`
2. **Nueva evaluación:** Clic en "Comenzar Evaluación"
3. **Completar formulario:** 8 parámetros clínicos
4. **Ver resultados:** Nivel de riesgo y recomendaciones
5. **Revisar historial:** Ver evaluaciones anteriores
6. **Consultar médico:** SIEMPRE con resultados

---

**¡Usa DiabPredict de manera responsable y siempre consulta con profesionales de la salud!** 🏥
