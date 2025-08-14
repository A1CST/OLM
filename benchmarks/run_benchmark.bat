@echo off
REM OLM System Benchmark Runner for Windows
REM This script helps run various benchmark scenarios

echo ============================================
echo    OLM System Performance Benchmark
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist results\benchmark_results.json (
    echo Installing benchmark dependencies...
    pip install -r benchmark_requirements.txt
)

echo Select benchmark mode:
echo.
echo 1. Quick Benchmark (2 minutes)
echo 2. Standard Benchmark (5 minutes)  
echo 3. Extended Benchmark (10 minutes)
echo 4. Interactive Mode (live monitoring)
echo 5. Custom duration
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo Running 2-minute benchmark...
    python benchmark.py --duration 120 --plots --output results\quick_benchmark.json
) else if "%choice%"=="2" (
    echo Running 5-minute benchmark...
    python benchmark.py --duration 300 --plots --output results\standard_benchmark.json
) else if "%choice%"=="3" (
    echo Running 10-minute benchmark...
    python benchmark.py --duration 600 --plots --output results\extended_benchmark.json
) else if "%choice%"=="4" (
    echo Starting interactive monitoring...
    python benchmark.py --interactive
) else if "%choice%"=="5" (
    set /p duration="Enter duration in seconds: "
    echo Running custom benchmark for !duration! seconds...
    python benchmark.py --duration !duration! --plots --output results\custom_benchmark.json
) else (
    echo Invalid choice. Running default 5-minute benchmark...
    python benchmark.py --duration 300 --plots
)

echo.
echo ============================================
echo Benchmark complete! Check the results files.
echo ============================================
pause