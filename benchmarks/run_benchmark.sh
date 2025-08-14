#!/bin/bash
# OLM System Benchmark Runner for Linux/macOS
# This script helps run various benchmark scenarios

echo "============================================"
echo "    OLM System Performance Benchmark"
echo "============================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found"
        echo "Please install Python 3.8+ to continue"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Install dependencies if needed
if [ ! -f "results/benchmark_results.json" ]; then
    echo "Installing benchmark dependencies..."
    pip install -r benchmark_requirements.txt
fi

echo
echo "Select benchmark mode:"
echo
echo "1. Quick Benchmark (2 minutes)"
echo "2. Standard Benchmark (5 minutes)"  
echo "3. Extended Benchmark (10 minutes)"
echo "4. Interactive Mode (live monitoring)"
echo "5. Custom duration"
echo

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo "Running 2-minute benchmark..."
        $PYTHON_CMD benchmark.py --duration 120 --plots --output results/quick_benchmark.json
        ;;
    2)
        echo "Running 5-minute benchmark..."
        $PYTHON_CMD benchmark.py --duration 300 --plots --output results/standard_benchmark.json
        ;;
    3)
        echo "Running 10-minute benchmark..."
        $PYTHON_CMD benchmark.py --duration 600 --plots --output results/extended_benchmark.json
        ;;
    4)
        echo "Starting interactive monitoring..."
        $PYTHON_CMD benchmark.py --interactive
        ;;
    5)
        read -p "Enter duration in seconds: " duration
        echo "Running custom benchmark for $duration seconds..."
        $PYTHON_CMD benchmark.py --duration $duration --plots --output results/custom_benchmark.json
        ;;
    *)
        echo "Invalid choice. Running default 5-minute benchmark..."
        $PYTHON_CMD benchmark.py --duration 300 --plots
        ;;
esac

echo
echo "============================================"
echo "Benchmark complete! Check the results files."
echo "============================================"