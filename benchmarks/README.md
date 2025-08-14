# OLM System Benchmarks

This directory contains all benchmarking tools and results for the Organic Learning Machine (OLM) system.

## Directory Structure

```
benchmarks/
├── benchmark.py           # Main benchmarking script
├── benchmark_requirements.txt  # Python dependencies for benchmarking
├── run_benchmark.bat      # Windows batch runner
├── run_benchmark.sh       # Unix/Linux shell runner
├── results/              # Benchmark results (JSON files)
├── plots/                # Generated performance plots
└── README.md             # This documentation
```

## Quick Start

### Method 1: Interactive Runner (Recommended)

**Windows:**
```cmd
cd benchmarks
run_benchmark.bat
```

**Linux/macOS:**
```bash
cd benchmarks
chmod +x run_benchmark.sh
./run_benchmark.sh
```

### Method 2: Direct Python Execution

```bash
cd benchmarks
python benchmark.py --duration 300 --plots --output results/my_benchmark.json
```

## Benchmark Modes

1. **Quick Benchmark (2 minutes)** - Fast performance snapshot
2. **Standard Benchmark (5 minutes)** - Comprehensive analysis  
3. **Extended Benchmark (10+ minutes)** - Long-term stability testing
4. **Interactive Mode** - Real-time monitoring with live statistics
5. **Custom Duration** - Specify your own monitoring duration

## Command Line Options

```bash
python benchmark.py [options]

Options:
  --duration SECONDS     Monitoring duration (default: indefinite)
  --interval SECONDS     Sampling interval (default: 1.0)
  --output FILENAME      Results JSON file (default: results/benchmark_results.json)
  --plots               Generate performance plots in plots/ directory
  --interactive         Live monitoring mode with real-time updates
```

## What Gets Monitored

### System Metrics
- **CPU Usage**: Overall and per-core utilization, frequency scaling
- **Memory Usage**: System RAM, process-specific allocation, swap usage  
- **GPU Monitoring**: Utilization, memory usage, temperature (NVIDIA GPUs)
- **CUDA Metrics**: Memory allocation, cache usage, OOM events
- **I/O Statistics**: Network and disk throughput
- **Temperature**: System thermal monitoring

### OLM-Specific Metrics
- Process memory consumption for all OLM components
- Engine performance correlation with resource usage
- Growth stage impact on system load
- Neural processing depth vs. resource requirements

## Output Files

### Results Directory (`results/`)
All JSON result files are saved here with comprehensive performance data:
- System information and configuration
- Time-series performance data
- Statistical summaries (min, max, mean, percentiles)
- OLM-specific process metrics

### Plots Directory (`plots/`)
When using `--plots` option, generates:
- `cpu_usage.png` - CPU utilization over time (overall + per-core)
- `memory_usage.png` - System and OLM process memory consumption
- `gpu_usage.png` - GPU utilization and memory allocation

## Dependencies

Install benchmark dependencies:
```bash
cd benchmarks
pip install -r benchmark_requirements.txt
```

Required packages:
- psutil (system monitoring)
- GPUtil (GPU monitoring, optional)
- matplotlib (plotting, optional)
- torch (CUDA monitoring, optional)

## Integration with OLM Engine

The benchmark tool can automatically detect and monitor running OLM processes. It tracks:
- `web_server.py` - Web interface process
- `engine_pytorch.py` - Main AI engine process
- `benchmark.py` - The benchmark process itself

## Example Usage

### Basic 5-minute benchmark with plots:
```bash
cd benchmarks
python benchmark.py --duration 300 --plots
```

### Interactive monitoring:
```bash
cd benchmarks  
python benchmark.py --interactive
```

### Custom benchmark with specific output:
```bash
cd benchmarks
python benchmark.py --duration 600 --output results/long_test.json --plots
```

## Troubleshooting

**Import Errors:** Make sure you're running from the `benchmarks` directory and that the parent directory contains the OLM engine files.

**GPU Monitoring Issues:** Install GPUtil and ensure NVIDIA drivers are properly installed.

**Permission Errors:** On Unix systems, make sure the shell script is executable: `chmod +x run_benchmark.sh`

## Performance Baselines

Typical resource usage for OLM system:
- **Idle State**: ~15-25% CPU, ~1-2GB RAM
- **Active Learning**: ~45-65% CPU, ~2-4GB RAM  
- **Heavy Processing**: ~70-90% CPU, ~4-6GB RAM
- **GPU Acceleration**: ~60-85% GPU, ~8-14GB VRAM

Results are automatically timestamped and can be compared across different runs to track performance improvements or regressions.