# Organic Learning Machine (OLM)

An autonomous AI agent that learns from experience through real-time sensory input, featuring adaptive neural processing, associative memory, and organic growth cycles.

## 🧠 Overview

The Organic Learning Machine is a sophisticated AI system that mimics biological learning processes. Unlike traditional AI models trained on static datasets, the OLM learns continuously from its environment through:

- **Real-time sensory input** (vision, mouse tracking, system monitoring)
- **Associative memory** using Locality-Sensitive Hashing
- **Adaptive neural processing** with dynamic depth selection
- **Life-cycle management** with sleep/wake cycles and growth stages
- **Multi-modal neural heads** for specialized processing tasks

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sensory Input │───▶│   Tokenization   │───▶│ Neural Pipeline │
│                 │    │                  │    │                 │
│ • Screen Capture│    │ Raw Data ───▶    │    │ • Pattern LSTM  │
│ • Mouse Tracking│    │ 142D Vector      │    │ • Compression   │
│ • System Stats  │    │                  │    │ • D-LSTM Heads  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                       │                       │
          │              ┌─────────────────┐             │
          │              │  LSH Memory     │             │
          │              │  System         │             │
          │              │                 │             │
          │              │ • Pattern Match │             │
          │              │ • Novelty Detect│             │
          │              │ • Depth Cache   │             │
          └──────────────│                 │─────────────┘
                         └─────────────────┘
                                  │
                    ┌─────────────────────────────┐
                    │      Regulator Module       │
                    │                             │
                    │ • Growth Stages             │
                    │ • State Management          │
                    │ • Energy/Boredom Tracking   │
                    │ • Sleep/Wake Cycles         │
                    └─────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **NVIDIA GPU** with CUDA support (tested on RTX 4080)
- **Python 3.8+**
- **CUDA 12.1+** and compatible PyTorch

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/A1CST/OLM.git
   cd OLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch with CUDA:** (Critical for performance)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Run CUDA setup (Windows):**
   ```bash
   setup_cuda_env.bat
   ```

5. **Start the system:**
   ```bash
   python web_server.py
   ```

6. **Open web interface:**
   Navigate to `http://localhost:5001`

## 💻 Web Interface

The system provides a comprehensive web dashboard with multiple specialized views:

### **Main Dashboard** (`/`)
- **Engine Control**: Start/Stop/Wipe system data
- **Live Status**: Real-time engine state, energy, active tokens
- **Lifetime Stats**: Total runtime, tick count, session history
- **System Logs**: Live system and error logs

### **Metrics Page** (`/metrics`)
- Neural head performance tracking
- Processing depth statistics
- Energy and boredom levels over time
- TPS (Ticks Per Second) monitoring

### **Memory Hashes** (`/hashes`)
- LSH memory system visualization
- Pattern frequency analysis
- Master hash database browser
- Memory promotion/demotion tracking

### **Interactive Page** (`/interactive`)
- Real-time sensory data visualization
- Live vision feed processing
- Mouse tracking display
- System resource monitoring

### **Statistics** (`/stats`)
- Long-term performance analytics
- Growth stage progression
- Session comparison tools
- Historical data export

### **Monitor** (`/monitor`)
- Real-time system health monitoring
- GPU utilization and temperature
- Memory and CPU usage
- Process monitoring

## 🧩 Core Components

### **1. Engine Core** (`engine_pytorch.py`)

The heart of the OLM system, implementing a state machine-based AI agent.

**Key Classes:**
- `EngineCore(threading.Thread)`: Main processing thread (20 TPS)

**Core Methods:**
- `run()`: Main state machine loop
- `_tick_cycle()`: Single processing cycle
- `_preprocess_vision_data()`: Visual input processing
- `_find_optimal_depth()`: Dynamic neural depth selection

**State Machine:**
- **AWAKE**: Active learning and processing
- **SLEEPING**: Memory consolidation and growth
- **CRYING**: Novelty-seeking behavior (future feature)

### **2. Regulator Module** (`regulator.py`)

Manages the agent's lifecycle, growth, and energy systems.

**Key Features:**
- **Growth Stages**: Agent matures from stage 1.0 upward
- **Dynamic Parameters**: Energy capacity, processing depth, sleep duration
- **State Decisions**: Determines when to sleep, wake, or seek novelty

**Growth Model:**
```python
# Energy capacity increases with growth
energy_cap = min(1.0, 0.4 + growth_stage * 0.05)

# Processing depth expands with maturity  
max_depth = int(min(physical_max_depth, 4 + growth_stage))

# Awake duration extends as agent matures
awake_duration = int(1200 + growth_stage * 500)  # ~1min to 10+min
```

### **3. Neural Pipeline** (`torch_models.py`)

Multi-stage PyTorch neural network architecture.

**Network Components:**
- **Pattern LSTM**: Initial sensory pattern recognition
- **Compression LSTM**: Dimensionality reduction and feature extraction
- **D-LSTM Networks**: Variable-depth processing (4-256 layers)
- **VAE Encoder**: Visual novelty detection (frozen CNN)

**Specialized Heads:**
- `internal_thought`: Core reasoning and decision making
- `speech`: Text generation using character vocabulary
- `video`/`image`: Visual processing and generation
- `action`: Motor control and action selection

### **4. Sensory Systems**

#### **Vision System** (`new_vision.py`)
- **Screen Capture**: Full desktop at configurable TPS
- **Multi-perspective Processing**: Full frame, ROI, periphery
- **Base64 Encoding**: Efficient data transmission
- **Mouse-following ROI**: Dynamic region of interest

#### **Mouse Tracking** (`mouse.py`)
- **Real-time Position**: 20Hz coordinate tracking
- **Event Detection**: Click, hold, release, scroll
- **Movement Analysis**: Delta calculations and velocity
- **Behavioral Patterns**: Click duration analysis

#### **System Monitoring** (`internals.py`)
- **Resource Tracking**: CPU, memory, GPU utilization
- **Temperature Monitoring**: System and GPU thermal data
- **Process Management**: Active process counting
- **Performance Metrics**: Load averages and system health

### **5. Memory Systems**

#### **Tokenizer** (`tokenizer.py`)
Converts raw sensory data into normalized 142-dimensional vectors.

**Vector Layout:**
- **Slots 0-9**: Mouse data (position, movement, clicks)
- **Slots 10-13**: System internals (CPU, memory, GPU, temp)
- **Slots 14-77**: Visual novelty vector (64D from VAE)
- **Slots 78-141**: Static latent representation (64D)

**Key Features:**
- **Adaptive Normalization**: Learning min/max ranges over time
- **Bidirectional Conversion**: Encoding and decoding support
- **Persistent Statistics**: Saved normalization parameters

#### **LSH Memory System** (`hashing.py`)
Locality-Sensitive Hashing for efficient pattern recognition and associative memory.

**Memory Architecture:**
- **Master Hashes**: Long-term memory (100 most frequent patterns)
- **Novel Cache**: Short-term memory for new patterns
- **Demoted Cache**: Recently removed patterns (50 items)

**Pattern Lifecycle:**
1. **Detection**: New patterns enter novel cache
2. **Promotion**: Patterns with 3+ occurrences promoted to master
3. **Optimization**: Stores optimal processing depths per pattern
4. **Similarity**: Hamming distance ≤ 5% for pattern matching

### **6. Action Selection** (`action_selector.py`)

Energy-efficient neural head activation system.

**Selection Algorithm:**
1. `internal_thought` always active (mandatory core reasoning)
2. Calculate energy budget (5% of current agent energy)
3. Rank heads by: `novelty_score × boredom_level`
4. Activate highest-priority heads within energy budget

### **7. Statistics Tracking** (`system_tracker.py`)

Persistent long-term memory for system performance.

**Tracked Metrics:**
- **Lifetime Statistics**: Total ticks, runtime, sessions
- **Session Details**: Start/end times, performance data
- **Growth History**: Agent development over time
- **Performance Analytics**: Processing efficiency metrics

## ⚙️ Configuration

### **Model Configuration**
```python
# Neural network heads configuration
head_configs = {
    "internal_thought": {"hidden_size": 64, "max_depth": 16},
    "speech": {"hidden_size": 64, "max_depth": 16, "output_size": vocab_size},
    "video": {"hidden_size": 64, "max_depth": 16, "output_size": 64},
    "image": {"hidden_size": 64, "max_depth": 16, "output_size": 64},
    "action": {"hidden_size": 128, "max_depth": 16, "output_size": action_vocab}
}
```

### **Processing Parameters**
- **Target TPS**: 20 (50ms per cycle)
- **Vision TPS**: 5 (200ms per frame)
- **LSH Dimensions**: 128 hash functions
- **Memory Limits**: 100 master patterns, 50 novel patterns

### **Growth Parameters**
```python
# Regulator thresholds (tunable hyperparameters)
sleep_energy_threshold = 0.2      # Sleep when energy drops below 20%
cry_boredom_threshold = 0.9       # Seek novelty when 90% bored
growth_rate = 0.1                 # Growth per novelty unit
```

## 📊 Data Flow

### **Processing Cycle (Every 50ms):**

1. **Sensory Collection**
   ```
   Vision → Mouse → System Stats → Raw Data Dictionary
   ```

2. **Tokenization**
   ```
   Raw Data → Adaptive Normalization → 142D Vector
   ```

3. **Memory Query**
   ```
   Vector → LSH Hashing → Pattern Similarity → Novelty Score
   ```

4. **Neural Processing**
   ```
   Vector → Pattern LSTM → Compression → D-LSTM Heads → Outputs
   ```

5. **Action Selection**
   ```
   Energy Budget → Head Ranking → Selective Activation
   ```

6. **State Management**
   ```
   Energy/Boredom Check → Regulator → State Transition
   ```

### **Sleep Cycle Processing:**
1. **Memory Consolidation**: Pattern frequency analysis
2. **Growth Update**: Novelty-based maturation
3. **Parameter Adjustment**: Dynamic threshold updates
4. **Data Persistence**: Save critical state information

## 📊 Performance Benchmarking

The OLM system includes a comprehensive benchmarking tool to monitor system performance, resource usage, and optimization opportunities.

### **Benchmark Tool** (`benchmark.py`)

A sophisticated performance monitoring system that tracks:

**System Metrics:**
- **CPU Usage**: Overall and per-core utilization, frequency scaling
- **Memory Usage**: System RAM, process-specific allocation, swap usage
- **GPU Monitoring**: Utilization, memory usage, temperature (NVIDIA GPUs)
- **CUDA Metrics**: Memory allocation, cache usage, OOM events
- **I/O Statistics**: Network and disk throughput
- **Temperature**: System thermal monitoring

**OLM-Specific Tracking:**
- Process memory consumption for all OLM components
- Engine performance correlation with resource usage
- Growth stage impact on system load
- Neural processing depth vs. resource requirements

### **Running Benchmarks**

**Quick Start:**
```bash
# Install benchmark dependencies
pip install -r benchmark_requirements.txt

# Run standard 5-minute benchmark
python benchmark.py --duration 300 --plots

# Interactive mode with live updates
python benchmark.py --interactive

# Use convenience scripts
./run_benchmark.sh          # Linux/macOS
run_benchmark.bat          # Windows
```

**Command Line Options:**
```bash
python benchmark.py [options]

Options:
  --duration SECONDS     Monitoring duration (default: indefinite)
  --interval SECONDS     Sampling interval (default: 1.0)
  --output FILENAME      Results JSON file (default: benchmark_results.json)
  --plots               Generate performance plots
  --interactive         Live monitoring mode
```

**Benchmark Modes:**
1. **Quick Benchmark** (2 minutes): Rapid performance snapshot
2. **Standard Benchmark** (5 minutes): Comprehensive analysis  
3. **Extended Benchmark** (10+ minutes): Long-term stability testing
4. **Interactive Mode**: Real-time monitoring with live statistics

### **Benchmark Output**

**JSON Results File:**
```json
{
  "system_info": {
    "cpu_count": 16,
    "total_ram_gb": 32.0,
    "cuda_devices": [{"name": "RTX 4080", "memory_total_mb": 16384}]
  },
  "performance_summary": {
    "cpu_percent": {"min": 12.5, "max": 89.3, "mean": 45.2},
    "ram_percent": {"min": 34.1, "max": 67.8, "mean": 52.4},
    "olm_memory_gb": {"min": 1.2, "max": 3.8, "mean": 2.1}
  }
}
```

**Generated Visualizations:**
- `cpu_usage.png`: CPU utilization over time (overall + per-core)
- `memory_usage.png`: System and OLM process memory consumption  
- `gpu_usage.png`: GPU utilization and memory allocation

**Live Statistics Display:**
```
🔥 LIVE OLM SYSTEM PERFORMANCE
============================================================
💻 CPU Usage:      45.2%
🧠 RAM Usage:      16.8GB (52.4%)
🤖 OLM Memory:     2.134GB
🎮 GPU Usage:      78.5% | 12.3GB
⚡ CUDA Memory:    11.247GB

📊 Last Minute Averages:
   CPU: 43.8% | RAM: 51.2%
```

### **Performance Analysis**

**Typical Resource Usage:**
- **Idle State**: ~15-25% CPU, ~1-2GB RAM
- **Active Learning**: ~45-65% CPU, ~2-4GB RAM  
- **Heavy Processing**: ~70-90% CPU, ~4-6GB RAM
- **GPU Acceleration**: ~60-85% GPU, ~8-14GB VRAM

**Optimization Insights:**
- Monitor GPU utilization for CUDA efficiency
- Track memory growth patterns for leak detection
- Analyze CPU spikes during neural processing
- Correlate performance with growth stage progression

**Benchmark Integration:**
```bash
# Run OLM engine and benchmark simultaneously
python web_server.py &                    # Start OLM in background
python benchmark.py --duration 600 --plots  # 10-minute benchmark

# Automated testing workflow
./run_benchmark.sh    # Select benchmark mode
# Results automatically saved with timestamp
```

## 🔧 Advanced Features

### **Dynamic Depth Selection**
The system automatically adjusts neural processing depth based on pattern complexity:

```python
# Depth search levels adapt to agent's growth stage
search_limit = min(physical_max_depth, regulator.current_max_depth_cap)
base_levels = [4, 8, 16, 32, 64, 128, 256]
depth_levels = [d for d in base_levels if d <= search_limit]
```

### **Organic Growth Model**
Agent capabilities expand naturally with experience:

- **Energy Capacity**: 40% → 100% as agent matures
- **Processing Depth**: 4 → 16+ layers available
- **Attention Span**: 1 minute → 10+ minutes awake time
- **Learning Rate**: Larger training samples with growth

### **Real-time Performance Monitoring**
- **TPS Tracking**: Maintains consistent 20 Hz processing
- **Memory Profiling**: RAM and GPU usage optimization
- **Thermal Management**: Temperature-aware processing
- **Error Recovery**: Graceful handling of system failures

## 🛠️ Development

### **File Structure**
```
OLM/
├── engine_pytorch.py      # Main processing engine
├── regulator.py          # Lifecycle management
├── web_server.py         # Flask web interface
├── torch_models.py       # PyTorch neural networks
├── tokenizer.py          # Sensory data processing
├── system_tracker.py     # Statistics and persistence
├── hashing.py            # LSH memory system
├── action_selector.py    # Neural head selection
├── new_vision.py         # Screen capture system
├── mouse.py              # Mouse tracking
├── internals.py          # System monitoring
├── requirements.txt      # Python dependencies
├── models/               # Saved weights and data
│   ├── vae_encoder_weights.pth
│   ├── pipeline_weights.pth
│   ├── tokenizer_checkpoint.json
│   ├── master_hashes.json
│   └── system_stats.json
├── static/               # Web interface assets
│   ├── main.js
│   ├── style.css
│   └── [specialized page assets]
└── templates/            # HTML templates
    ├── index.html
    ├── metrics.html
    ├── hashes.html
    ├── interactive.html
    ├── stats.html
    └── monitor.html
```

### **Adding New Sensory Inputs**
1. **Extend Tokenizer**: Add slots in 142D vector
2. **Update Collection**: Add data source to sensory collection
3. **Modify Normalization**: Add adaptive scaling for new features
4. **Test Integration**: Verify vector consistency

### **Creating New Neural Heads**
1. **Define Configuration**: Add to `head_configs` dictionary
2. **Implement Decoder**: Add decoding logic to tokenizer
3. **Update Action Selector**: Include in energy budgeting
4. **Add Visualization**: Create web interface components

### **Memory System Extensions**
1. **Hash Function Tuning**: Adjust LSH parameters for new data
2. **Pattern Promotion**: Modify promotion criteria
3. **Storage Optimization**: Implement compression for large patterns
4. **Retrieval Enhancement**: Add semantic similarity measures

## 🔍 Troubleshooting

### **Common Issues**

**CUDA Not Detected:**
```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**High CPU Usage:**
- Ensure CUDA is properly configured
- Reduce vision capture frequency
- Check for memory leaks in long-running sessions

**Web Interface Not Loading:**
- Verify port 5001 is available
- Check firewall settings
- Ensure Flask and SocketIO versions are compatible

**Memory Errors:**
- Reduce batch sizes in neural processing
- Clear old model checkpoints
- Monitor GPU memory usage

### **Performance Optimization**

**GPU Utilization:**
- Batch processing for efficiency
- Memory pooling for frequent allocations
- Mixed precision training (future feature)

**Memory Management:**
- Regular garbage collection
- Efficient tensor operations
- Smart caching strategies

**Processing Speed:**
- Vectorized operations where possible
- Async I/O for file operations
- Optimized image processing pipelines

## 📈 Future Enhancements

### **Planned Features**
- **Dream State Processing**: Enhanced memory consolidation during sleep
- **Multi-Agent Communication**: Distributed OLM networks
- **Reinforcement Learning**: Action-outcome association learning
- **Natural Language Interface**: Direct conversation capabilities
- **Mobile Deployment**: Lightweight inference versions

### **Research Directions**
- **Continual Learning**: Avoiding catastrophic forgetting
- **Meta-Learning**: Learning to learn more efficiently
- **Causal Reasoning**: Understanding cause and effect
- **Self-Modification**: Agent-directed architecture changes

## 🤝 Contributing

We welcome contributions to the Organic Learning Machine project! Areas of particular interest:

- **Neural Architecture Improvements**: More efficient processing designs
- **Memory System Enhancements**: Better pattern recognition and storage
- **Sensory Input Expansion**: New types of environmental data
- **Performance Optimization**: Faster processing and lower resource usage
- **Documentation**: Improved guides and API documentation

### **Development Guidelines**
1. **Code Style**: Follow existing patterns and conventions
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update README and inline comments
4. **Performance**: Profile changes for efficiency impact

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Flask/SocketIO**: For enabling real-time web interfaces  
- **Scientific Python Community**: NumPy, SciPy, PIL, and related libraries
- **CUDA/NVIDIA**: For GPU acceleration capabilities

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Report bugs and request features](https://github.com/A1CST/OLM/issues)
- **Discussions**: [Community discussions and questions](https://github.com/A1CST/OLM/discussions)

---

**Built with 🧠 by the OLM Development Team**

*Advancing the frontier of autonomous learning systems*