@echo off
chcp 65001 >nul
echo ============================================
echo   安装依赖库 numpy scipy
echo ============================================
echo.

F:\Qt\python.exe -m pip install numpy scipy

echo.
echo ============================================
echo   运行 beamforming.py
echo ============================================
echo.

F:\Qt\python.exe "C:\Users\Lenovo\Documents\xwechat_files\wxid_s5wf0viiu4kd22_f76e\msg\file\2026-07\beamforming.py"

echo.
echo ============================================
echo   运行结束
echo ============================================
pause
