"""
OLM System Benchmark Tool

A comprehensive performance monitoring and benchmarking tool for the 
Organic Learning Machine (OLM) system. Monitors CPU, RAM, GPU usage,
and provides detailed performance analytics while the engine is running.

Usage:
    python benchmark.py [--duration 300] [--interval 1] [--output benchmark_results.json]
"""

import psutil
import time
import json
import argparse
import threading
import subprocess
import sys
import os
import queue
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not installed. GPU monitoring will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. CUDA monitoring disabled.")

try:
    from engine_pytorch import EngineCore
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("Warning: engine_pytorch not available. Cannot auto-start engine.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization disabled.")


class SystemBenchmark:
    """
    Comprehensive system resource monitoring for OLM engine performance analysis.
    
    Tracks:
    - CPU usage (overall and per-core)
    - RAM usage (system and process-specific)
    - GPU utilization and memory
    - CUDA memory allocation
    - Process-specific metrics
    - Temperature monitoring
    - Network and disk I/O
    """
    
    def __init__(self, interval=1.0, max_history=3600, start_engine=True):
        """
        Initialize benchmark monitoring.
        
        Args:
            interval (float): Sampling interval in seconds
            max_history (int): Maximum number of samples to keep in memory
            start_engine (bool): Whether to automatically start the OLM engine
        """
        self.interval = interval
        self.max_history = max_history
        self.running = False
        self.start_time = None
        self.start_engine = start_engine
        
        # Engine management
        self.engine = None
        self.engine_queue = None
        
        # Data storage
        self.data = defaultdict(lambda: deque(maxlen=max_history))
        self.timestamps = deque(maxlen=max_history)
        
        # Process tracking
        self.target_processes = ['python', 'Python']  # Track Python processes
        self.olm_processes = []
        
        # GPU detection
        self.gpu_available = GPU_AVAILABLE
        self.cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
        # System info
        self.system_info = self._get_system_info()
        
        print(f"[INIT] Benchmark initialized:")
        print(f"   - Sampling interval: {interval}s")
        print(f"   - Max history: {max_history} samples")
        print(f"   - GPU monitoring: {'YES' if self.gpu_available else 'NO'}")
        print(f"   - CUDA monitoring: {'YES' if self.cuda_available else 'NO'}")
        print(f"   - Engine auto-start: {'YES' if self.start_engine and ENGINE_AVAILABLE else 'NO'}")
        print(f"   - System: {self.system_info['cpu_count']} CPU cores, {self.system_info['total_ram_gb']:.1f}GB RAM")
    
    def _get_system_info(self):
        """Collect static system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'total_ram_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform,
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.cuda_available:
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_devices'] = []
            for i in range(torch.cuda.device_count()):
                info['cuda_devices'].append({
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_mb': torch.cuda.get_device_properties(i).total_memory / (1024**2)
                })
        
        return info
    
    def _find_olm_processes(self):
        """Find and track OLM-related processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
            try:
                # Look for Python processes running OLM files
                if proc.info['name'] in self.target_processes:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if any(olm_file in cmdline for olm_file in ['web_server.py', 'engine_pytorch.py', 'benchmark.py']):
                        processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        self.olm_processes = processes
        return len(processes)
    
    def _start_olm_engine(self):
        """Start the OLM engine for benchmarking."""
        if not ENGINE_AVAILABLE:
            print("[WARNING] Engine not available. Cannot start OLM engine.")
            return False
        
        if self.engine and self.engine.is_alive():
            print("[INFO] Engine already running.")
            return True
            
        try:
            print("[ENGINE] Starting OLM engine...")
            # Create queue for engine updates (similar to web_server.py)
            self.engine_queue = queue.Queue(maxsize=5)
            
            # Create and start engine
            self.engine = EngineCore(self.engine_queue)
            self.engine.start()
            
            # Wait a moment for engine to initialize
            time.sleep(2)
            
            if self.engine.is_alive():
                print("[ENGINE] OLM engine started successfully!")
                return True
            else:
                print("[ERROR] Failed to start OLM engine.")
                return False
                
        except Exception as e:
            print(f"[ERROR] Exception starting engine: {e}")
            return False
    
    def _stop_olm_engine(self):
        """Stop the OLM engine."""
        if self.engine and self.engine.is_alive():
            print("[ENGINE] Stopping OLM engine...")
            try:
                self.engine.stop()
                # Give it a moment to stop gracefully
                time.sleep(1)
                print("[ENGINE] OLM engine stopped.")
            except Exception as e:
                print(f"[WARNING] Exception stopping engine: {e}")
        else:
            print("[INFO] Engine not running.")
    
    def _collect_cpu_metrics(self):
        """Collect detailed CPU utilization metrics."""
        # Overall CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.data['cpu_percent'].append(cpu_percent)
        
        # Per-core usage
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        for i, usage in enumerate(cpu_per_core):
            self.data[f'cpu_core_{i}'].append(usage)
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self.data['cpu_freq_current'].append(cpu_freq.current)
                self.data['cpu_freq_max'].append(cpu_freq.max)
        except:
            pass
        
        # Load averages (Unix-like systems)
        try:
            load_avg = os.getloadavg()
            self.data['load_avg_1min'].append(load_avg[0])
            self.data['load_avg_5min'].append(load_avg[1])
            self.data['load_avg_15min'].append(load_avg[2])
        except:
            pass
    
    def _collect_memory_metrics(self):
        """Collect memory usage statistics."""
        # System memory
        memory = psutil.virtual_memory()
        self.data['ram_total_gb'].append(memory.total / (1024**3))
        self.data['ram_used_gb'].append(memory.used / (1024**3))
        self.data['ram_available_gb'].append(memory.available / (1024**3))
        self.data['ram_percent'].append(memory.percent)
        
        # Swap memory
        swap = psutil.swap_memory()
        self.data['swap_total_gb'].append(swap.total / (1024**3))
        self.data['swap_used_gb'].append(swap.used / (1024**3))
        self.data['swap_percent'].append(swap.percent)
        
        # Process-specific memory
        olm_memory_total = 0
        olm_process_count = self._find_olm_processes()
        
        # Include our own engine if running
        if self.engine and self.engine.is_alive():
            try:
                # Get the actual engine process
                engine_pid = os.getpid()  # Our process contains the engine thread
                engine_proc = psutil.Process(engine_pid)
                memory_info = engine_proc.memory_info()
                olm_memory_total += memory_info.rss
                olm_process_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        for proc in self.olm_processes:
            try:
                memory_info = proc.memory_info()
                olm_memory_total += memory_info.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        self.data['olm_memory_gb'].append(olm_memory_total / (1024**3))
        self.data['olm_process_count'].append(olm_process_count)
    
    def _collect_gpu_metrics(self):
        """Collect GPU utilization and memory statistics."""
        if not self.gpu_available:
            return
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                self.data[f'gpu_{i}_utilization'].append(gpu.load * 100)
                self.data[f'gpu_{i}_memory_used_gb'].append(gpu.memoryUsed / 1024)
                self.data[f'gpu_{i}_memory_total_gb'].append(gpu.memoryTotal / 1024)
                self.data[f'gpu_{i}_memory_percent'].append(gpu.memoryUtil * 100)
                self.data[f'gpu_{i}_temperature'].append(gpu.temperature)
        except Exception as e:
            print(f"Warning: GPU metrics collection failed: {e}")
    
    def _collect_cuda_metrics(self):
        """Collect CUDA-specific metrics."""
        if not self.cuda_available:
            return
        
        try:
            for i in range(torch.cuda.device_count()):
                # Memory allocation
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                
                self.data[f'cuda_{i}_memory_allocated_gb'].append(allocated)
                self.data[f'cuda_{i}_memory_cached_gb'].append(cached)
                
                # Memory stats if available
                try:
                    stats = torch.cuda.memory_stats(i)
                    self.data[f'cuda_{i}_alloc_retries'].append(stats.get('num_alloc_retries', 0))
                    self.data[f'cuda_{i}_ooms'].append(stats.get('num_ooms', 0))
                except:
                    pass
                    
        except Exception as e:
            print(f"Warning: CUDA metrics collection failed: {e}")
    
    def _collect_io_metrics(self):
        """Collect network and disk I/O statistics."""
        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            self.data['net_bytes_sent'].append(net_io.bytes_sent)
            self.data['net_bytes_recv'].append(net_io.bytes_recv)
            self.data['net_packets_sent'].append(net_io.packets_sent)
            self.data['net_packets_recv'].append(net_io.packets_recv)
        except:
            pass
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.data['disk_bytes_read'].append(disk_io.read_bytes)
                self.data['disk_bytes_write'].append(disk_io.write_bytes)
        except:
            pass
    
    def _collect_temperature_metrics(self):
        """Collect system temperature data."""
        try:
            temps = psutil.sensors_temperatures()
            for name, entries in temps.items():
                for i, entry in enumerate(entries):
                    key = f'temp_{name}_{i}' if len(entries) > 1 else f'temp_{name}'
                    self.data[key].append(entry.current)
        except:
            pass
    
    def _collect_all_metrics(self):
        """Collect all system metrics in one sampling cycle."""
        timestamp = datetime.now()
        self.timestamps.append(timestamp)
        
        # Collect all metrics
        self._collect_cpu_metrics()
        self._collect_memory_metrics()
        self._collect_gpu_metrics()
        self._collect_cuda_metrics()
        self._collect_io_metrics()
        self._collect_temperature_metrics()
    
    def start_monitoring(self, duration=None):
        """
        Start continuous system monitoring.
        
        Args:
            duration (float): Duration in seconds to monitor (None for indefinite)
        """
        baseline_duration = getattr(self, 'baseline_duration', 10)
        
        # Collect baseline measurements before starting engine
        if self.start_engine:
            print(f"[BASELINE] Collecting baseline measurements for {baseline_duration} seconds...")
            print("   - This establishes performance before engine starts")
            
            # Start baseline monitoring
            self.running = True
            self.start_time = time.time()
            baseline_end = self.start_time + baseline_duration
            baseline_samples = 0
            
            # Collect baseline data
            try:
                while time.time() < baseline_end and self.running:
                    cycle_start = time.time()
                    self._collect_all_metrics()
                    baseline_samples += 1
                    
                    # Maintain sampling interval
                    cycle_time = time.time() - cycle_start
                    sleep_time = max(0, self.interval - cycle_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
            except KeyboardInterrupt:
                print("\n[STOP] Baseline collection stopped by user")
                self.running = False
                return self._generate_summary()
            
            # Show baseline summary
            self._print_baseline_summary(baseline_samples)
        
        # Start OLM engine if requested
        engine_started = False
        if self.start_engine:
            engine_started = self._start_olm_engine()
            if not engine_started:
                print("[WARNING] Continuing benchmark without engine...")
                # Reset for monitoring without engine
                self.running = True
                self.start_time = time.time()
            else:
                # Give engine time to warm up and start processing
                warmup_time = getattr(self, 'warmup_time', 5)
                print(f"[ENGINE] Warming up for {warmup_time} seconds...")
                time.sleep(warmup_time)
                
                # Reset timer for main monitoring phase
                self.start_time = time.time()
        else:
            # No engine start, just begin monitoring
            self.running = True
            self.start_time = time.time()
        
        end_time = self.start_time + duration if duration else None
        
        print(f"[START] Starting benchmark monitoring...")
        print(f"   - Duration: {'Indefinite' if duration is None else f'{duration}s'}")
        print(f"   - Engine running: {'YES' if engine_started else 'NO'}")
        print(f"   - Press Ctrl+C to stop manually\n")
        
        sample_count = 0
        
        try:
            while self.running:
                cycle_start = time.time()
                
                # Collect metrics
                self._collect_all_metrics()
                sample_count += 1
                
                # Print periodic status
                if sample_count % 60 == 0:  # Every 60 samples (1 minute at 1Hz)
                    elapsed = time.time() - self.start_time
                    self._print_status_update(elapsed, sample_count)
                
                # Check duration limit
                if end_time and time.time() >= end_time:
                    break
                
                # Maintain sampling interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.interval - cycle_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[STOP] Monitoring stopped by user")
        
        self.running = False
        elapsed = time.time() - self.start_time
        print(f"\n[DONE] Monitoring completed: {sample_count} samples in {elapsed:.1f}s")
        
        # Stop engine if we started it
        if self.start_engine:
            self._stop_olm_engine()
        
        return self._generate_summary()
    
    def _print_status_update(self, elapsed, sample_count):
        """Print periodic status updates during monitoring."""
        if not self.data['cpu_percent']:
            return
            
        current_cpu = self.data['cpu_percent'][-1]
        current_ram = self.data['ram_percent'][-1]
        
        status = f"[TIME] {elapsed/60:.1f}m elapsed | Samples: {sample_count} | "
        status += f"CPU: {current_cpu:.1f}% | RAM: {current_ram:.1f}%"
        
        if self.data.get('olm_memory_gb'):
            olm_mem = self.data['olm_memory_gb'][-1]
            status += f" | OLM: {olm_mem:.2f}GB"
        
        if self.gpu_available and self.data.get('gpu_0_utilization'):
            gpu_util = self.data['gpu_0_utilization'][-1]
            status += f" | GPU: {gpu_util:.1f}%"
        
        print(status)
    
    def _print_baseline_summary(self, baseline_samples):
        """Print summary of baseline measurements before engine starts."""
        if not self.data['cpu_percent'] or len(self.data['cpu_percent']) < baseline_samples:
            print("[WARNING] Insufficient baseline data collected")
            return
            
        # Calculate baseline averages
        cpu_baseline = list(self.data['cpu_percent'])[-baseline_samples:]
        ram_baseline = list(self.data['ram_percent'])[-baseline_samples:]
        
        baseline_cpu_avg = sum(cpu_baseline) / len(cpu_baseline)
        baseline_ram_avg = sum(ram_baseline) / len(ram_baseline)
        
        print(f"\n[BASELINE] System performance before engine:")
        print(f"   - CPU Usage:  {baseline_cpu_avg:5.1f}% average")
        print(f"   - RAM Usage:  {baseline_ram_avg:5.1f}% average")
        
        if self.gpu_available and len(self.data.get('gpu_0_utilization', [])) >= baseline_samples:
            gpu_baseline = list(self.data['gpu_0_utilization'])[-baseline_samples:]
            baseline_gpu_avg = sum(gpu_baseline) / len(gpu_baseline)
            print(f"   - GPU Usage:  {baseline_gpu_avg:5.1f}% average")
        
        # Store baseline for comparison
        self.baseline_stats = {
            'cpu_avg': baseline_cpu_avg,
            'ram_avg': baseline_ram_avg,
            'gpu_avg': sum(list(self.data.get('gpu_0_utilization', []))[-baseline_samples:]) / baseline_samples if self.gpu_available and len(self.data.get('gpu_0_utilization', [])) >= baseline_samples else 0,
            'samples': baseline_samples
        }
        print()
    
    def _generate_summary(self):
        """Generate performance summary statistics."""
        if not self.timestamps:
            return {"error": "No data collected"}
        
        summary = {
            'monitoring_info': {
                'start_time': self.start_time,
                'duration_seconds': time.time() - self.start_time,
                'sample_count': len(self.timestamps),
                'sampling_interval': self.interval
            },
            'system_info': self.system_info,
            'performance_summary': {}
        }
        
        # Calculate statistics for key metrics
        key_metrics = [
            'cpu_percent', 'ram_percent', 'ram_used_gb', 'olm_memory_gb',
            'gpu_0_utilization', 'gpu_0_memory_percent',
            'cuda_0_memory_allocated_gb'
        ]
        
        for metric in key_metrics:
            if metric in self.data and self.data[metric]:
                values = list(self.data[metric])
                summary['performance_summary'][metric] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return summary
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to JSON file."""
        results = {
            'system_info': self.system_info,
            'monitoring_config': {
                'interval': self.interval,
                'max_history': self.max_history,
                'start_time': self.start_time,
                'duration': time.time() - self.start_time if self.start_time else 0
            },
            'summary': self._generate_summary(),
            'raw_data': {
                'timestamps': [ts.isoformat() for ts in self.timestamps],
                'metrics': {k: list(v) for k, v in self.data.items()}
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[SAVE] Results saved to: {filename}")
        print(f"   - File size: {os.path.getsize(filename) / 1024:.1f} KB")
        return filename
    
    def generate_plots(self, output_dir="benchmark_plots"):
        """Generate visualization plots of the benchmark data."""
        if not MATPLOTLIB_AVAILABLE:
            print("[ERROR] Matplotlib not available. Cannot generate plots.")
            return
        
        if not self.timestamps:
            print("[ERROR] No data to plot.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert timestamps for plotting
        times = [mdates.date2num(ts) for ts in self.timestamps]
        
        # CPU Usage Plot
        self._create_cpu_plot(times, output_dir)
        
        # Memory Usage Plot  
        self._create_memory_plot(times, output_dir)
        
        # GPU Usage Plot (if available)
        if self.gpu_available:
            self._create_gpu_plot(times, output_dir)
        
        print(f"[PLOTS] Plots saved to: {output_dir}/")
    
    def _create_cpu_plot(self, times, output_dir):
        """Create CPU utilization plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Overall CPU usage
        if 'cpu_percent' in self.data:
            ax1.plot(times, list(self.data['cpu_percent']), 'b-', linewidth=2, label='Overall CPU')
            ax1.set_ylabel('CPU Usage (%)')
            ax1.set_title('OLM System CPU Utilization')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Per-core usage
        core_count = self.system_info['cpu_count']
        colors = plt.cm.tab10(range(min(core_count, 10)))
        
        for i in range(min(core_count, 8)):  # Limit to first 8 cores for readability
            if f'cpu_core_{i}' in self.data:
                ax2.plot(times, list(self.data[f'cpu_core_{i}']), 
                        color=colors[i], alpha=0.7, label=f'Core {i}')
        
        ax2.set_ylabel('CPU Usage (%)')
        ax2.set_xlabel('Time')
        ax2.set_title('Per-Core CPU Utilization')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cpu_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_memory_plot(self, times, output_dir):
        """Create memory usage plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # System memory
        if 'ram_used_gb' in self.data:
            ax1.plot(times, list(self.data['ram_used_gb']), 'g-', linewidth=2, label='System RAM Used')
            ax1.plot(times, list(self.data['ram_total_gb']), 'g--', alpha=0.5, label='Total RAM')
        
        ax1.set_ylabel('Memory (GB)')
        ax1.set_title('System Memory Usage')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Process-specific memory
        if 'olm_memory_gb' in self.data:
            ax2.plot(times, list(self.data['olm_memory_gb']), 'r-', linewidth=2, label='OLM Processes')
        
        ax2.set_ylabel('Memory (GB)')
        ax2.set_xlabel('Time')
        ax2.set_title('OLM Process Memory Usage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_gpu_plot(self, times, output_dir):
        """Create GPU utilization and memory plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # GPU utilization
        if 'gpu_0_utilization' in self.data:
            ax1.plot(times, list(self.data['gpu_0_utilization']), 'purple', linewidth=2, label='GPU Utilization')
        
        ax1.set_ylabel('GPU Usage (%)')
        ax1.set_title('GPU Utilization')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # GPU and CUDA memory
        if 'gpu_0_memory_used_gb' in self.data:
            ax2.plot(times, list(self.data['gpu_0_memory_used_gb']), 'orange', linewidth=2, label='GPU Memory Used')
        
        if 'cuda_0_memory_allocated_gb' in self.data:
            ax2.plot(times, list(self.data['cuda_0_memory_allocated_gb']), 'red', linewidth=2, label='CUDA Allocated')
        
        ax2.set_ylabel('Memory (GB)')
        ax2.set_xlabel('Time')
        ax2.set_title('GPU Memory Usage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gpu_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_live_stats(self):
        """Print real-time statistics (for interactive mode)."""
        if not self.data['cpu_percent']:
            print("No data collected yet...")
            return
        
        print("\n" + "="*60)
        print("[LIVE] OLM SYSTEM PERFORMANCE")
        print("="*60)
        
        # Current values
        cpu = self.data['cpu_percent'][-1]
        ram_gb = self.data['ram_used_gb'][-1] 
        ram_pct = self.data['ram_percent'][-1]
        
        print(f"CPU Usage:     {cpu:6.1f}%")
        print(f"RAM Usage:     {ram_gb:6.2f}GB ({ram_pct:.1f}%)")
        
        if 'olm_memory_gb' in self.data:
            olm_mem = self.data['olm_memory_gb'][-1]
            print(f"OLM Memory:    {olm_mem:6.3f}GB")
        
        if self.gpu_available and 'gpu_0_utilization' in self.data:
            gpu_util = self.data['gpu_0_utilization'][-1]
            gpu_mem = self.data.get('gpu_0_memory_used_gb', [0])[-1]
            print(f"GPU Usage:     {gpu_util:6.1f}% | {gpu_mem:.2f}GB")
        
        if self.cuda_available and 'cuda_0_memory_allocated_gb' in self.data:
            cuda_mem = self.data['cuda_0_memory_allocated_gb'][-1]
            print(f"CUDA Memory:   {cuda_mem:6.3f}GB")
        
        # Performance averages
        if len(self.data['cpu_percent']) > 60:  # Last minute average
            recent_cpu = list(self.data['cpu_percent'])[-60:]
            recent_ram = list(self.data['ram_percent'])[-60:]
            
            print(f"\n[AVG] Last Minute Averages:")
            print(f"   CPU: {statistics.mean(recent_cpu):5.1f}% | RAM: {statistics.mean(recent_ram):5.1f}%")


def run_interactive_mode():
    """Run benchmark in interactive mode with live updates."""
    benchmark = SystemBenchmark(interval=1.0)
    
    print("[INTERACTIVE] Interactive Mode - Live Performance Monitoring")
    print("Press 's' to save results, 'p' to generate plots, 'q' to quit\n")
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=benchmark.start_monitoring, daemon=True)
    monitor_thread.start()
    
    try:
        while benchmark.running:
            time.sleep(5)  # Update every 5 seconds
            benchmark.print_live_stats()
            
            # Simple command interface
            print("\n[s]ave | [p]lots | [q]uit | [Enter] to continue...")
            # Note: This is simplified - a full implementation would use
            # non-blocking input or a proper UI framework
            
    except KeyboardInterrupt:
        benchmark.running = False
        
    print("\n[SAVE] Saving final results...")
    benchmark.save_results()
    benchmark.generate_plots()


def main():
    """Main benchmark program entry point."""
    parser = argparse.ArgumentParser(
        description="OLM System Performance Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                              # Interactive mode
  python benchmark.py --duration 300               # 5-minute benchmark
  python benchmark.py --interval 0.5 --output my_benchmark.json
  python benchmark.py --duration 600 --plots      # 10-min with plots
        """
    )
    
    parser.add_argument('--duration', type=float, default=None,
                       help='Monitoring duration in seconds (default: indefinite)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output filename for results (default: benchmark_results.json)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate performance plots after monitoring')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode with live updates')
    parser.add_argument('--no-engine', action='store_true',
                       help='Skip auto-starting the OLM engine (monitor system only)')
    parser.add_argument('--engine-warmup', type=int, default=5,
                       help='Seconds to wait after starting engine before monitoring (default: 5)')
    parser.add_argument('--baseline-duration', type=int, default=10,
                       help='Seconds to collect baseline before starting engine (default: 10)')
    
    args = parser.parse_args()
    
    # Header
    print("OLM System Performance Benchmark Tool")
    print("="*50)
    
    if args.interactive:
        run_interactive_mode()
        return
    
    # Standard benchmark mode
    start_engine = not args.no_engine
    benchmark = SystemBenchmark(interval=args.interval, start_engine=start_engine)
    benchmark.warmup_time = args.engine_warmup
    benchmark.baseline_duration = args.baseline_duration
    
    try:
        summary = benchmark.start_monitoring(duration=args.duration)
        
        # Save results
        benchmark.save_results(args.output)
        
        # Generate plots if requested
        if args.plots:
            benchmark.generate_plots()
        
        # Print summary
        print("\n[SUMMARY] BENCHMARK RESULTS")
        print("="*50)
        
        perf = summary.get('performance_summary', {})
        baseline = getattr(benchmark, 'baseline_stats', None)
        
        if baseline:
            print("                   Baseline  |  With Engine  |  Difference")
            print("-"*50)
            
            if 'cpu_percent' in perf:
                cpu_stats = perf['cpu_percent']
                cpu_diff = cpu_stats['mean'] - baseline['cpu_avg']
                print(f"CPU Usage:     {baseline['cpu_avg']:6.1f}%   | {cpu_stats['mean']:7.1f}%   | +{cpu_diff:5.1f}%")
            
            if 'ram_percent' in perf:
                ram_stats = perf['ram_percent']
                ram_diff = ram_stats['mean'] - baseline['ram_avg']
                print(f"RAM Usage:     {baseline['ram_avg']:6.1f}%   | {ram_stats['mean']:7.1f}%   | +{ram_diff:5.1f}%")
            
            if 'gpu_0_utilization' in perf and baseline['gpu_avg'] > 0:
                gpu_stats = perf['gpu_0_utilization']
                gpu_diff = gpu_stats['mean'] - baseline['gpu_avg']
                print(f"GPU Usage:     {baseline['gpu_avg']:6.1f}%   | {gpu_stats['mean']:7.1f}%   | +{gpu_diff:5.1f}%")
            
            if 'olm_memory_gb' in perf:
                olm_stats = perf['olm_memory_gb']
                print(f"OLM Memory:    {0:6.2f}GB  | {olm_stats['mean']:7.2f}GB  | +{olm_stats['mean']:5.2f}GB")
                
        else:
            # No baseline, show standard summary
            if 'cpu_percent' in perf:
                cpu_stats = perf['cpu_percent']
                print(f"CPU Usage:     Avg: {cpu_stats['mean']:5.1f}% | Max: {cpu_stats['max']:5.1f}%")
            
            if 'ram_percent' in perf:
                ram_stats = perf['ram_percent']
                print(f"RAM Usage:     Avg: {ram_stats['mean']:5.1f}% | Max: {ram_stats['max']:5.1f}%")
            
            if 'olm_memory_gb' in perf:
                olm_stats = perf['olm_memory_gb']
                print(f"OLM Memory:    Avg: {olm_stats['mean']:5.2f}GB | Max: {olm_stats['max']:5.2f}GB")
            
            if 'gpu_0_utilization' in perf:
                gpu_stats = perf['gpu_0_utilization']
                print(f"GPU Usage:     Avg: {gpu_stats['mean']:5.1f}% | Max: {gpu_stats['max']:5.1f}%")
        
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        return 1
    
    print(f"\n[SUCCESS] Benchmark complete! Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())