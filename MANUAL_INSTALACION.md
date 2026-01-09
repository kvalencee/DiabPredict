# 📖 Manual de Instalación - DiabPredict

## 🎯 Opciones de Instalación

Tienes **dos opciones** para usar DiabPredict:

1. **Opción A (RECOMENDADA)**: Ejecutar con Python - Más fácil y rápido
2. **Opción B (OPCIONAL)**: Crear ejecutable standalone - Para distribuir sin Python

---

## 📦 Opción A: Instalación con Python (RECOMENDADA)

### Requisitos Previos
- ✅ Python 3.10 o superior ([Descargar](https://www.python.org/downloads/))
- ✅ Conexión a Internet (solo para instalación inicial)

### Paso 1: Verificar Python
```bash
python --version
```
Debe mostrar: `Python 3.10.x` o superior

### Paso 2: Navegar al Directorio del Proyecto
```bash
cd "c:\Users\Kevin Valencia\Documents\ESCOM\diabpredictt\DiabPredict"
```

### Paso 3: Instalar Dependencias (si no lo has hecho)
```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar la Aplicación
```bash
python run.py
```

**¡Listo!** La aplicación se abrirá automáticamente en tu navegador en `http://127.0.0.1:5000`

### Para Detener la Aplicación
Presiona `CTRL + C` en la terminal

### Para Ejecutar Nuevamente
```bash
cd "c:\Users\Kevin Valencia\Documents\ESCOM\diabpredictt\DiabPredict"
python run.py
```

---

## 🚀 Opción B: Crear Ejecutable Standalone (OPCIONAL)

Esta opción crea un archivo `.exe` que puede ejecutarse sin tener Python instalado.

### ⚠️ Nota Importante
El ejecutable será grande (~150-200 MB) porque incluye Python y todas las librerías.

### Paso 1: Instalar PyInstaller
```bash
pip install pyinstaller
```

### Paso 2: Crear el Ejecutable

**Opción 2A - Usando el script automático:**
```bash
cd "c:\Users\Kevin Valencia\Documents\ESCOM\diabpredictt\DiabPredict"
python -m PyInstaller --name=DiabPredict --onefile --windowed --add-data="app/templates;app/templates" --add-data="app/static;app/static" --add-data="ml/models;ml/models" --hidden-import=sklearn.utils._weight_vector --hidden-import=sklearn.neighbors._partition_nodes --collect-all=sklearn run.py
```

**Opción 2B - Comando simplificado:**
```bash
pyinstaller --onefile --windowed --name=DiabPredict run.py
```

### Paso 3: Ubicar el Ejecutable
El archivo `DiabPredict.exe` estará en:
```
DiabPredict\dist\DiabPredict.exe
```

### Paso 4: Copiar Archivos Necesarios
Para que el ejecutable funcione, copia junto a él:
- Carpeta `ml/models/` (con los archivos .pkl)
- Carpeta `app/templates/`
- Carpeta `app/static/`

### Paso 5: Ejecutar
Doble clic en `DiabPredict.exe`

---

## 🔧 Solución de Problemas

### Problema: "ModuleNotFoundError"
**Solución:**
```bash
pip install -r requirements.txt
```

### Problema: "Port 5000 is already in use"
**Solución:** Detén otros procesos en el puerto 5000 o cambia el puerto en `config.py`

### Problema: PyInstaller no funciona
**Solución:** Usa la Opción A (ejecutar con Python directamente)

### Problema: El ejecutable no encuentra los modelos
**Solución:** Asegúrate de copiar la carpeta `ml/models/` junto al ejecutable

---

## 📋 Verificación de Instalación

### Checklist
- [ ] Python 3.10+ instalado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Dataset descargado (`data/raw/diabetes.csv` existe)
- [ ] Modelos entrenados (archivos `.pkl` en `ml/models/`)
- [ ] Aplicación ejecuta sin errores (`python run.py`)
- [ ] Navegador abre en `http://127.0.0.1:5000`

---

## 🎓 Recomendación Final

**Para uso personal o desarrollo:** Usa la **Opción A** (Python directo)
- ✅ Más fácil de mantener
- ✅ Actualizaciones más rápidas
- ✅ Menos espacio en disco

**Para distribución a otros usuarios:** Usa la **Opción B** (Ejecutable)
- ✅ No requiere Python instalado
- ✅ Más fácil para usuarios finales
- ⚠️ Archivo grande (~150-200 MB)

---

## 📞 Soporte

Si tienes problemas:
1. Verifica que Python 3.10+ esté instalado
2. Asegúrate de estar en el directorio correcto
3. Revisa que todas las dependencias estén instaladas
4. Consulta la sección de solución de problemas arriba

**¡Disfruta usando DiabPredict!** 🎉
