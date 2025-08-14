
#!/usr/bin/env python3
"""
System Internals Monitor - Comprehensive Computer Statistics
==========================================================

A console-based system monitoring application that captures ALL possible
computer statistics on a 20Hz tick cycle (50ms per tick).
Updates statistics every 4 ticks (200ms).

Usage:
    python internals.py

Target: 20 TPS (50ms per tick)
Stats Update: Every 4 ticks (every 200ms)
"""

import time
import argparse
import threading
import psutil
import platform
import os
import sys
from datetime import datetime

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available - GPU monitoring disabled")

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False
    print("WMI not available - some temperature monitoring disabled")

class SystemMonitor:
    """
    Comprehensive system monitoring that runs on a 20Hz cycle (50ms per tick).
    Updates all statistics every 4 ticks.
    """
    
    def __init__(self, quiet=False, log_every=20):
        self.is_running = False
        self.tick_count = 0
        self.start_time = None
        self.last_tick_time = None
        
        # Monitoring settings
        self.update_interval = 4  # Update stats every 4 ticks
        self.tick_times = []  # Store last 20 tick times for TPS calculation
        self.last_tps_update = 0
        self.tps_update_interval = 1.0  # Update TPS display every second
        
        # Logging control
        self.quiet = quiet
        self.log_every = max(1, int(log_every))
        
        # System information
        self.system_info = self._get_system_info()
        
        # Initialize WMI for temperature monitoring (Windows)
        self.wmi_connection = None
        if WMI_AVAILABLE and platform.system() == "Windows":
            try:
                self.wmi_connection = wmi.WMI()
            except Exception as e:
                print(f"WMI connection failed: {e}")
        
        # Statistics storage
        self.current_stats = {}
        
        # Setup signal handler for graceful shutdown
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C for graceful shutdown."""
        print("\n\nReceived interrupt signal. Shutting down...")
        self.stop()
        sys.exit(0)
    
    def _get_system_info(self):
        """Get basic system information."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'hostname': platform.node(),
            'python_version': platform.python_version(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _get_cpu_stats(self):
        """Get comprehensive CPU statistics."""
        try:
            return {
                'cpu_percent_overall': psutil.cpu_percent(interval=0.1)
            }
        except Exception as e:
            print(f"Error getting CPU stats: {e}")
            return {}
    
    def _get_memory_stats(self):
        """Get comprehensive memory statistics."""
        try:
            virtual_memory = psutil.virtual_memory()
            return {
                'memory_used': virtual_memory.used,
                'memory_percent': virtual_memory.percent
            }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}
    
    def _get_disk_stats(self):
        """Get comprehensive disk statistics."""
        return {}
    
    def _get_network_stats(self):
        """Get comprehensive network statistics."""
        return {}
    
    def _get_gpu_stats(self):
        """Get GPU statistics using GPUtil."""
        if not GPU_AVAILABLE:
            return {}
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_stats = []
            
            for gpu in gpus:
                gpu_stats.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100 if gpu.load else 0,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    'temperature': gpu.temperature,
                    'uuid': gpu.uuid
                })
            
            return {'gpus': gpu_stats}
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return {}
    
    def _get_temperature_stats(self):
        """Get temperature statistics using WMI (Windows) and sensors."""
        try:
            temps = {}
            
            # WMI temperature monitoring (Windows)
            if self.wmi_connection:
                try:
                    # CPU temperature
                    cpu_temps = self.wmi_connection.Win32_PerfFormattedData_Counters_ThermalZoneInformation()
                    for temp in cpu_temps:
                        temps['cpu_wmi'] = temp.Temperature
                    
                    # GPU temperature
                    gpu_temps = self.wmi_connection.Win32_PerfFormattedData_Counters_GPUPerformanceCounters()
                    for temp in gpu_temps:
                        temps['gpu_wmi'] = temp.Temperature
                        
                except Exception as e:
                    pass
            
            # Try to get temperatures from psutil (if available)
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    sensor_temps = psutil.sensors_temperatures()
                    for name, entries in sensor_temps.items():
                        for entry in entries:
                            temps[f'{name}_{entry.label}'] = entry.current
            except Exception:
                pass
            
            return temps
        except Exception as e:
            print(f"Error getting temperature stats: {e}")
            return {}
    
    def _get_process_stats(self):
        """Get process statistics."""
        try:
            return {
                'total_processes': len(list(psutil.process_iter()))
            }
        except Exception as e:
            print(f"Error getting process stats: {e}")
            return {}
    
    def _get_system_load(self):
        """Get system load average."""
        try:
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                return {
                    'load_1min': load_avg[0],
                    'load_5min': load_avg[1],
                    'load_15min': load_avg[2]
                }
            return {}
        except Exception as e:
            print(f"Error getting system load: {e}")
            return {}
    
    def _get_battery_stats(self):
        """Get battery statistics."""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    'battery_percent': battery.percent,
                    'battery_power_plugged': battery.power_plugged,
                    'battery_time_left': battery.secsleft if battery.secsleft != -1 else None
                }
            return {}
        except Exception as e:
            print(f"Error getting battery stats: {e}")
            return {}
    
    def _collect_all_stats(self):
        """Collect all system statistics."""
        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'system_info': self.system_info,
            'cpu': self._get_cpu_stats(),
            'memory': self._get_memory_stats(),
            'disk': self._get_disk_stats(),
            'network': self._get_network_stats(),
            'gpu': self._get_gpu_stats(),
            'temperature': self._get_temperature_stats(),
            'processes': self._get_process_stats(),
            'system_load': self._get_system_load(),
            'battery': self._get_battery_stats()
        }
        
        return stats
    

    
    def _print_compact_stats(self, stats):
        """Print a compact, single-line-per-topic summary suitable for Engine parsing when quiet."""
        # CPU overall usage
        cpu = stats.get('cpu', {})
        if cpu:
            overall = cpu.get('cpu_percent_overall')
            if overall is not None:
                print(f"Overall Usage: {overall:.1f}%")

        # Memory used and percent
        mem = stats.get('memory', {})
        if mem:
            used = mem.get('memory_used', 0)
            percent = mem.get('memory_percent', 0)
            # format used as GB with 1 decimal
            used_gb = used / (1024**3)
            print(f"Used: {used_gb:.1f} GB ({percent:.1f}%)")

        # GPU load (first GPU if available)
        gpu = stats.get('gpu', {})
        gpus = gpu.get('gpus') if gpu else None
        if gpus:
            load = gpus[0].get('load')
            if load is not None:
                print(f"Load: {load:.1f}%")

        # Temperature (print one representative sensor if available)
        temps = stats.get('temperature', {})
        if temps:
            # Pick any numeric temperature value
            for _, val in temps.items():
                try:
                    t = float(val)
                    print(f"Temperature: {t:.0f}°C")
                    break
                except Exception:
                    continue

        # System load averages
        load = stats.get('system_load', {})
        if load:
            l1 = load.get('load_1min')
            l5 = load.get('load_5min')
            l15 = load.get('load_15min')
            if l1 is not None and l5 is not None and l15 is not None:
                print(f"1min: {l1:.2f} | 5min: {l5:.2f} | 15min: {l15:.2f}")

        # Battery percent
        bat = stats.get('battery', {})
        if bat and bat.get('battery_percent') is not None:
            print(f"Level: {bat['battery_percent']:.1f}%")

        # Total processes
        proc = stats.get('processes', {})
        total_proc = proc.get('total_processes')
        if total_proc is not None:
            print(f"Total Processes: {total_proc}")

    def _calculate_tps(self):
        """Calculate current ticks per second."""
        if len(self.tick_times) < 2:
            return 0.0
        
        time_span = self.tick_times[-1] - self.tick_times[0]
        if time_span > 0:
            return (len(self.tick_times) - 1) / time_span
        return 0.0
    
    def start(self):
        """Start the system monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.last_tick_time = self.start_time
        
        if not self.quiet:
            print("=" * 100)
            print("SYSTEM INTERNALS MONITOR - Comprehensive Computer Statistics")
            print("=" * 100)
            print("Starting 20Hz tick cycle (50ms per tick)")
            print("Target: 20 TPS")
            print(f"Stats Update: Every {self.update_interval} ticks (every {self.update_interval * 50}ms)")
            print("Press Ctrl+C to exit")
            print("=" * 100)
            print()
        
        # Start the tick cycle
        self._tick_cycle()
    
    def stop(self):
        """Stop the system monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 100)
        print("SYSTEM MONITORING STOPPED")
        print("=" * 100)
        print(f"Total ticks: {self.tick_count}")
        print(f"Total time: {total_time:.2f} seconds")
        if total_time > 0:
            print(f"Average TPS: {self.tick_count / total_time:.2f}")
        print("=" * 100)
    
    def _tick_cycle(self):
        """Main tick cycle function - called every 50ms (20Hz)."""
        if not self.is_running:
            return
        
        # TICK CYCLE START
        current_time = time.time()
        self.tick_count += 1
        
        # Calculate tick timing
        if self.last_tick_time:
            tick_duration = current_time - self.last_tick_time
        else:
            tick_duration = 0
        
        self.last_tick_time = current_time
        
        # Collect and display stats every N ticks
        if self.tick_count % self.update_interval == 0:
            stats = self._collect_all_stats()
            self._print_compact_stats(stats)
        
        # Track tick times for TPS calculation
        self.tick_times.append(current_time)
        
        # Keep only last 20 ticks for TPS calculation
        if len(self.tick_times) > 20:
            self.tick_times.pop(0)
        
        # TICK CYCLE END - Schedule next tick in 50ms (20Hz)
        if self.is_running:
            timer = threading.Timer(0.050, self._tick_cycle)
            timer.daemon = True
            timer.start()

def main():
    """Main entry point."""
    try:
        parser = argparse.ArgumentParser(description='System Internals Monitor')
        parser.add_argument('--quiet', action='store_true', help='Reduce console output')
        parser.add_argument('--log-every', type=int, default=20, help='If quiet, log every N ticks')
        args = parser.parse_args()

        # Create and start the system monitor
        monitor = SystemMonitor(quiet=args.quiet, log_every=args.log_every)
        
        # Start monitoring
        monitor.start()
        
        # Keep the main thread alive
        while monitor.is_running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 