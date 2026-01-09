@echo off
echo ===================================
echo   Empaquetando DiabPredict
echo   (Version App de Escritorio)
echo ===================================
echo.

REM Limpiar builds anteriores
echo Limpiando builds anteriores...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo Empaquetando con PyInstaller (modo app de escritorio)...
echo.

python -m PyInstaller --name=DiabPredict ^
            --onefile ^
            --windowed ^
            --noconsole ^
            --add-data="app/templates;app/templates" ^
            --add-data="app/static;app/static" ^
            --add-data="ml/models;ml/models" ^
            --add-data="data/processed;data/processed" ^
            --hidden-import=sklearn.utils._weight_vector ^
            --hidden-import=sklearn.neighbors._partition_nodes ^
            --hidden-import=werkzeug.security ^
            --hidden-import=jinja2 ^
            --collect-all=sklearn ^
            --noconfirm ^
            run_desktop.py

echo.
if exist dist\DiabPredict.exe (
    echo ===================================
    echo   Empaquetado completado
    echo ===================================
    echo.
    echo Ejecutable generado en: dist\DiabPredict.exe
    echo.
    echo CARACTERISTICAS:
    echo   - NO muestra consola
    echo   - Abre navegador automaticamente
    echo   - Se comporta como app de escritorio
    echo   - Tamanio: ~75 MB
    echo.
    echo NOTA: El ejecutable incluye todas las dependencias
    echo       y NO requiere Python instalado.
    echo.
) else (
    echo ===================================
    echo   Error en el empaquetado
    echo ===================================
    echo.
    echo Revisa los mensajes de error arriba.
    echo.
)

pause
