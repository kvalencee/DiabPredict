"""
Punto de entrada principal para DiabPredict
"""
import os
import sys
import webbrowser
from threading import Timer
from app import create_app


def open_browser(url):
    """Abre el navegador después de un pequeño delay"""
    webbrowser.open(url)


def main():
    """Función principal"""
    # Obtener configuración desde variable de entorno
    config_name = os.environ.get('FLASK_ENV', 'development')

    # Crear aplicación
    app = create_app(config_name)

    # Configuración del servidor
    host = app.config['HOST']
    port = app.config['PORT']
    debug = app.config['DEBUG']

    print("=" * 70)
    print(" DIABPREDICT - Sistema de Predicción de Riesgo de Diabetes Tipo 2")
    print("=" * 70)
    print(f"\n✓ Aplicación iniciada exitosamente")
    print(f"✓ Servidor ejecutándose en: http://{host}:{port}")
    print(f"✓ Modo: {config_name}")

    if not debug:
        print(f"\n→ Abriendo navegador...")
        # Abrir navegador automáticamente después de 1.5 segundos
        Timer(1.5, open_browser, args=[f"http://{host}:{port}"]).start()

    print(f"\nPresione CTRL+C para detener el servidor")
    print("=" * 70 + "\n")

    try:
        # Iniciar servidor Flask
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug  # Solo recargar automáticamente en desarrollo
        )
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("✓ Servidor detenido correctamente")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error al iniciar el servidor: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()