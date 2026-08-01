@echo off
chcp 65001 >nul
set "PYTHON310=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"

if not exist "%PYTHON310%" (
    echo Python 3.10 was not found at:
    echo %PYTHON310%
    echo.
    echo Please install Python 3.10 or update PYTHON310 in this bat file.
    pause
    exit /b 1
)

"%PYTHON310%" "%~dp0convert_csv_to_wav.py" --input-dir "%~dp08.1\separated" --sample-rate 44100

echo.
echo Finished. WAV files are in:
echo %~dp08.1\separated
pause
