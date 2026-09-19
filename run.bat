@echo off
title Dual Hand Finger Counter
cd /d "%~dp0"

:menu
cls
echo =================================================================
echo   Dual-Hand Tracking and Finger Counter
echo =================================================================
echo.
echo   [1] Launch Web Dashboard (Streamlit Portal)
echo   [2] Launch Live Webcam Window (30+ FPS Real-time)
echo   [3] Run Automated Verification Tests
echo   [4] Exit
echo.
echo =================================================================
set /p choice="Select an option (1-4): "

if "%choice%"=="1" goto web_portal
if "%choice%"=="2" goto live_cam
if "%choice%"=="3" goto test_suite
if "%choice%"=="4" exit
goto menu

:web_portal
cls
echo Launching Web Dashboard...
streamlit run app.py
pause
goto menu

:live_cam
cls
echo Launching Live Camera Window (Press 'q' to quit, 'p' to save snapshot)...
python live_cam.py
pause
goto menu

:test_suite
cls
echo Running Automated Verification Tests...
python test_counter.py
pause
goto menu
