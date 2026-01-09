"""
Script para descargar el dataset de diabetes de Pima Indians
"""
import os
import urllib.request
from pathlib import Path

# URLs del dataset
KAGGLE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
BACKUP_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names"

# Directorio de destino
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / 'raw'
OUTPUT_FILE = RAW_DIR / 'diabetes.csv'

# Nombres de columnas
COLUMN_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]


def download_dataset():
    """Descarga el dataset de diabetes"""
    print("=" * 70)
    print("DESCARGANDO DATASET PIMA INDIANS DIABETES")
    print("=" * 70)
    
    # Crear directorio si no existe
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\n[*] Descargando desde: {KAGGLE_URL}")
        print(f"[*] Guardando en: {OUTPUT_FILE}")
        
        # Descargar el archivo
        urllib.request.urlretrieve(KAGGLE_URL, OUTPUT_FILE)
        
        # Agregar encabezados al archivo
        print("\n[OK] Dataset descargado exitosamente")
        print("[OK] Agregando nombres de columnas...")
        
        # Leer el contenido
        with open(OUTPUT_FILE, 'r') as f:
            content = f.read()
        
        # Escribir con encabezados
        with open(OUTPUT_FILE, 'w') as f:
            f.write(','.join(COLUMN_NAMES) + '\n')
            f.write(content)
        
        print("[OK] Encabezados agregados")
        
        # Verificar el archivo
        file_size = OUTPUT_FILE.stat().st_size
        print(f"\n[INFO] Informacion del dataset:")
        print(f"  - Tamanio: {file_size:,} bytes")
        print(f"  - Ubicacion: {OUTPUT_FILE}")
        
        # Contar líneas
        with open(OUTPUT_FILE, 'r') as f:
            lines = f.readlines()
        
        print(f"  - Filas totales: {len(lines) - 1} (+ 1 encabezado)")
        print(f"  - Columnas: {len(COLUMN_NAMES)}")
        
        print("\n" + "=" * 70)
        print("DESCARGA COMPLETADA EXITOSAMENTE")
        print("=" * 70)
        print("\nPuedes proceder a entrenar los modelos con:")
        print("  python ml/scripts/train_models.py")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error al descargar el dataset: {e}")
        print("\n[INFO] Solucion alternativa:")
        print("  1. Descarga manualmente desde:")
        print("     https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        print(f"  2. Guarda el archivo como: {OUTPUT_FILE}")
        return False


if __name__ == '__main__':
    success = download_dataset()
    exit(0 if success else 1)
