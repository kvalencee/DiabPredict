"""
Punto de entrada para el ejecutable de DiabPredict (versión app de escritorio)
"""
import os
import sys
import webbrowser
import threading
from time import sleep

def run_app():
    """Ejecuta la aplicación Flask en modo silencioso"""
    # Suprimir salida de Flask
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Importar y crear app
    from app import create_app
    app = create_app('production')
    
    # Ejecutar servidor en thread separado
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

def open_browser_delayed():
    """Abre el navegador después de que el servidor esté listo"""
    sleep(2)  # Esperar a que el servidor inicie
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Iniciar servidor en thread de fondo
    server_thread = threading.Thread(target=run_app, daemon=True)
    server_thread.start()
    
    # Abrir navegador
    open_browser_delayed()
    
    # Mantener la aplicación viva
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        sys.exit(0)
