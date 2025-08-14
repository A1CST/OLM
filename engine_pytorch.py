
import time
import threading
import queue
import torch
import numpy as np
import json
import os
import re
import subprocess
import sys
from collections import deque
from PIL import Image
import base64
from io import BytesIO
import torchvision.transforms as T

# Import the PyTorch models you created
from torch_models import ModelPipeline, VAE_Encoder
from regulator import Regulator
# Import the new tokenizer
from tokenizer import SensoryTokenizer
# Import the system statistics tracker
from system_tracker import SystemTracker
# Import the LSH system for associative memory
from hashing import LSHSystem
# Import the action selector for dynamic head selection
from action_selector import ActionSelector
# We'll add the VAE later, for now we import the pipeline


class EngineCore(threading.Thread):
    def __init__(self, update_queue: queue.Queue):
        super().__init__()
        self.daemon = True
        self.update_queue = update_queue

        # --- System Statistics Tracker ---
        self.tracker = SystemTracker()

        # --- Tokenizer Setup ---
        self.tokenizer = SensoryTokenizer()

        # --- LSH System (Associative Memory) ---
        self.hasher = LSHSystem(input_dim=self.tokenizer.vector_size, num_hashes=128)
        self._load_master_hashes() # Method to be added

        # --- Device and Model Setup ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Explicitly check for CUDA and log the status for debugging
        if torch.cuda.is_available():
            self.log("✅ CUDA is available. Engine will run on the GPU.")
            self.log(f"   - GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            self.log("⚠️ WARNING: CUDA not available. Engine will run on the CPU.")
            self.log("   - This will result in high CPU usage and low performance.")
            self.log("   - Please ensure PyTorch is installed with CUDA support and that you have a compatible NVIDIA GPU.")
        
        self.log(f"Engine device officially set to: {self.device.upper()}")
        
        # --- VAE Encoder Setup (Fixed Random Projector) ---
        self.vae_encoder = VAE_Encoder(latent_dim=64).to(self.device)
        vae_weights_path = 'models/vae_encoder_weights.pth'

        if os.path.exists(vae_weights_path):
            # If weights exist, load them
            self.vae_encoder.load_state_dict(torch.load(vae_weights_path, map_location=self.device))
            self.log("✅ Loaded existing VAE encoder weights.")
        else:
            # If this is the very first run, save the initial random weights
            self.log("⚠️ No VAE weights found. Saving initial random weights for future runs.")
            os.makedirs(os.path.dirname(vae_weights_path), exist_ok=True)
            torch.save(self.vae_encoder.state_dict(), vae_weights_path)
            
        # Freeze the encoder's parameters and set to evaluation mode
        self.vae_encoder.eval()
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        
        # Variables for the novelty vector calculation
        self.previous_latent_vector = None

        # --- Define model configurations ---
        self.tokenizer_dim = 142
        p_lstm_hidden = 256
        c_lstm_hidden = 128
        
        # Get vocabulary sizes from the tokenizer
        speech_vocab_size = len(self.tokenizer.char_map)
        action_vocab_size = len(self.tokenizer.action_map)
        latent_vector_size = 64 # A chosen size for image/video vectors

        # --- UPDATE THIS DICTIONARY ---
        head_configs = {
            "internal_thought": {"hidden_size": 64, "max_depth": 16}, # No specific output size
            "speech": {"hidden_size": 64, "max_depth": 16, "output_size": speech_vocab_size},
            "video": {"hidden_size": 64, "max_depth": 16, "output_size": latent_vector_size},
            "image": {"hidden_size": 64, "max_depth": 16, "output_size": latent_vector_size},
            "action": {"hidden_size": 128, "max_depth": 16, "output_size": action_vocab_size}
        }
        
        # --- ADD THIS BLOCK ---
        # Get the physical max depth from the config to pass to the regulator
        physical_max_depth = head_configs['internal_thought']['max_depth']
        self.regulator = Regulator(physical_max_depth=physical_max_depth)
        self.log("✅ Regulator module initialized.")
        # --- END OF BLOCK ---

        # --- NOW INITIALIZE THE ACTION SELECTOR ---
        self.action_selector = ActionSelector(head_configs)
        # Initialize novelty scores for all heads
        self.head_novelty_scores = {}
        for head_name in head_configs.keys():
            self.head_novelty_scores[head_name] = 1.0 # Start with a baseline score

        # Instantiate the PyTorch Model Pipeline and move it to the GPU
        self.pipeline = ModelPipeline(
            self.tokenizer_dim, 
            p_lstm_hidden, 
            c_lstm_hidden, 
            head_configs
        )
        self.pipeline.to(self.device)
        self.log("Model pipeline loaded onto device.")
        
        # --- ADD THIS LINE ---
        self._load_pipeline_weights()

        # --- State Management ---
        # This will hold the hidden/cell states for all LSTMs
        self.pipeline_states = self._initialize_pipeline_states(p_lstm_hidden, c_lstm_hidden, head_configs)
        
        # Start in stopped state - will be set to True when start() is called
        self.is_running = False
        
        # --- ADD THESE LINES ---
        # State machine attributes
        self.agent_state = "AWAKE" # Start in the AWAKE state
        self.ticks_spent_awake = 0
        # --- END OF BLOCK ---
        
        self.tick_count = 0
        self.start_time = None
        self.last_tps_update = time.time()
        self.tps = 0.0
        
        # TPS calculation variables
        self.tps_window_start = time.time()
        self.tps_window_ticks = 0
        
        # --- Component System Setup ---
        self.sensory_info = {}
        self.component_processes = {}
        
        # --- Action Selection and Credit Assignment ---
        # Buffer to link actions at tick N to the novelty change at tick N+1
        self.action_outcome_buffer = deque(maxlen=5)
        
        # Component configuration
        self.config = {
            'components': {
                'audio': {'enabled': False, 'script': 'tick_counter.py', 'args': ['--quiet', '--prefix', '[MIC]']},
                'audio_system': {'enabled': False, 'script': 'tick_counter.py', 'args': ['--quiet', '--loopback', '--prefix', '[SYS]']},
                'vision': {'enabled': True, 'script': 'new_vision.py', 'args': ['--headless', '--quiet']},
                'internals': {'enabled': True, 'script': 'internals.py', 'args': ['--quiet']},
                'mouse': {'enabled': True, 'script': 'mouse.py', 'args': ['--quiet']}
            }
        }
        
        # Start enabled components
        self._start_components()

    def log(self, message):
        """Helper function to send log messages to the GUI."""
        try:
            self.update_queue.put_nowait({'type': 'log', 'message': message})
        except queue.Full:
            try:
                self.update_queue.get_nowait()
                self.update_queue.put_nowait({'type': 'log', 'message': message})
            except queue.Empty:
                pass

    def _initialize_pipeline_states(self, p_size, c_size, heads_config):
        """Creates the initial hidden/cell states for all LSTMs."""
        # Note: LSTM hidden state is a tuple: (h_0, c_0)
        # h_0 shape: (num_layers, batch_size, hidden_size)
        # We use batch_size=1 and num_layers=1 for our LSTMs
        
        states = {
            'pattern': (torch.zeros(1, 1, p_size).to(self.device), torch.zeros(1, 1, p_size).to(self.device)),
            'compression': (torch.zeros(1, 1, c_size).to(self.device), torch.zeros(1, 1, c_size).to(self.device)),
            'heads': {}
        }
        for name, config in heads_config.items():
            h_size = config['hidden_size']
            m_depth = config['max_depth']
            # For DLSTM, we need a list of states for each layer
            h_list = [torch.zeros(1, 1, h_size).to(self.device) for _ in range(m_depth)]
            c_list = [torch.zeros(1, 1, h_size).to(self.device) for _ in range(m_depth)]
            states['heads'][name] = {'h': h_list, 'c': c_list}
            
        self.log("Initialized pipeline states.")
        return states

    def get_energy(self):
        """Returns the current energy level of the agent (0.0 to 1.0)."""
        # Simple energy model: starts at 1.0 and decreases with processing
        # Energy recovers over time when not processing heavily
        base_energy = 1.0
        processing_penalty = len(self.head_novelty_scores) * 0.01  # Small penalty per active head
        time_recovery = min(1.0, (time.time() - self.start_time) * 0.001)  # Gradual recovery
        
        energy = max(0.1, base_energy - processing_penalty + time_recovery)
        return energy

    def get_boredom(self):
        """Returns the current boredom level of the agent (0.0 to 1.0)."""
        # Boredom increases when the same patterns are repeated
        # It decreases when novel patterns are encountered
        base_boredom = 0.5
        novelty_factor = 1.0 - (len(self.hasher.master_hash_data) / 100.0)  # Less novelty as more patterns are learned
        repetition_penalty = 0.1  # Small penalty for repeated patterns
        
        boredom = min(1.0, base_boredom + novelty_factor + repetition_penalty)
        return boredom

    def _start_components(self):
        """Start all enabled components."""
        for comp_name, comp_config in self.config['components'].items():
            if comp_config['enabled']:
                self._start_component(comp_name, comp_config)
    
    def _start_component(self, comp_name, comp_config):
        """Start a single component process."""
        try:
            script_path = os.path.join(os.path.dirname(__file__), comp_config['script'])
            if not os.path.exists(script_path):
                self.log(f"⚠️ Component script not found: {script_path}")
                return
            
            cmd = [sys.executable, script_path] + comp_config['args']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.component_processes[comp_name] = process
            self.log(f"✅ Started component: {comp_name}")
            
            # Start a thread to read output from this component
            reader_thread = threading.Thread(
                target=self._read_component_output,
                args=(comp_name, process),
                daemon=True
            )
            reader_thread.start()
            
        except Exception as e:
            self.log(f"[ERROR] Failed to start component {comp_name}: {e}")
    
    def _read_component_output(self, comp_name, process):
        """Read output from a component process and parse it."""
        try:
            for line in iter(process.stdout.readline, ''):
                if not self.is_running:
                    break
                
                line = line.strip()
                if line:
                    self._ingest_component_line(comp_name, line)
                    
        except Exception as e:
            self.log(f"[ERROR] Error reading from {comp_name}: {e}")
        finally:
            process.terminate()
    
    def _ingest_component_line(self, name, line):
        """Parses a line of output from a component and updates sensory_info."""
        try:
            # --- Initialize dictionary if it doesn't exist ---
            if name not in self.sensory_info:
                self.sensory_info[name] = {}

            # --- Parser for 'mouse' component ---
            if name == 'mouse':
                # Example: "Mouse: X: 123, Y: 456, Delta: 7.89, Events: left_single_click|scroll_up_3"
                match = re.search(r"X:\s*(\d+),\s*Y:\s*(\d+),\s*Delta:\s*([\d.]+),\s*Events:\s*(.+)", line)
                if match:
                    self.sensory_info[name]['x'] = int(match.group(1))
                    self.sensory_info[name]['y'] = int(match.group(2))
                    self.sensory_info[name]['delta'] = float(match.group(3))
                    
                    events_str = match.group(4)
                    events = events_str.split('|') if events_str != 'none' else []
                    
                    # Store boolean flags for easy tokenization
                    self.sensory_info[name]['left_single_click'] = 1.0 if 'left_single_click' in events else 0.0
                    self.sensory_info[name]['right_single_click'] = 1.0 if 'right_single_click' in events else 0.0
                    self.sensory_info[name]['left_hold'] = 1.0 if 'left_hold' in events else 0.0
                    
                    # Parse scroll events like "scroll_up_5" or "scroll_down_3"
                    scroll_value = 0
                    for event in events:
                        if event.startswith('scroll_up'):
                            scroll_value = int(event.split('_')[-1])
                        elif event.startswith('scroll_down'):
                            scroll_value = -int(event.split('_')[-1])
                    self.sensory_info[name]['scroll'] = scroll_value
            
            # --- Advanced Parser for 'vision' component ---
            elif name == 'vision':
                if line.startswith('#BEGIN_VISION_FEATURES#'):
                    self.sensory_info[name]['is_parsing'] = True
                    self.sensory_info[name]['buffer'] = ""
                elif line.startswith('#END_VISION_FEATURES#'):
                    if self.sensory_info[name].get('is_parsing'):
                        # Process the buffered data here if needed, or just mark as done
                        pass
                    self.sensory_info[name]['is_parsing'] = False
                elif self.sensory_info[name].get('is_parsing'):
                    try:
                        key, value = line.split(':', 1)
                        # Store the base64 images, we don't need the other text features for now
                        if key in ['roi_image_base64', 'periphery_image_base64', 'display_image_base64']:
                            self.sensory_info[name][key] = value
                    except ValueError:
                        pass # Ignore lines without a ':'
                else:
                    # Handle individual image lines (fallback for current format)
                    if line.startswith('roi_image_base64:'):
                        b64_string = line.split(':', 1)[1]
                        self.sensory_info[name]['roi_image_b64'] = b64_string
                    elif line.startswith('display_image_base64:'):
                        b64_string = line.split(':', 1)[1]
                        self.sensory_info[name]['display_image_b64'] = b64_string
                    elif line.startswith('periphery_image_base64:'):
                        b64_string = line.split(':', 1)[1]
                        self.sensory_info[name]['periphery_image_b64'] = b64_string
                    elif line.startswith('mouse_position:'):
                        pos_string = line.split(':', 1)[1]
                        self.sensory_info[name]['mouse_position'] = pos_string
                    elif line.startswith('roi_bounds:'):
                        bounds_string = line.split(':', 1)[1]
                        self.sensory_info[name]['roi_bounds'] = bounds_string
            
            # --- Parser for 'internals' component ---
            elif name == 'internals':
                # Examples: "Overall Usage: 15.2%", "Load: 25.3%", "Temperature: 45°C"
                match = re.search(r"([a-zA-Z\s]+):\s*([\d.]+)%?", line)
                if match:
                    key_raw = match.group(1).strip().lower().replace(' ', '_')
                    value = float(match.group(2))
                    
                    # Clean up keys for consistency
                    key_map = {
                        'overall_usage': 'cpu_percent',
                        'used': 'memory_percent', # We'll grab the percent from the "Used: X GB (Y %)" line
                        'load': 'gpu_load',
                        'temperature': 'temperature_c'
                    }
                    # Special case for memory line
                    mem_match = re.search(r"Used:.*?\((\d+\.\d+)%\)", line)
                    if mem_match:
                        self.sensory_info[name]['memory_percent'] = float(mem_match.group(1))
                    elif key_raw in key_map:
                         self.sensory_info[name][key_map[key_raw]] = value

        except Exception as e:
            self.log(f"[ERROR] Failed to parse line from {name}: {e}")
    
    def _preprocess_vision_data(self, vision_data):
        """
        Decodes, converts, and combines ROI and Periphery images into a single tensor.
        Returns: A single torch.Tensor for the VAE, or None if data is missing.
        """
        roi_b64 = vision_data.get('roi_image_base64')
        periphery_b64 = vision_data.get('periphery_image_b64')

        if not roi_b64 or not periphery_b64:
            return None

        try:
            # Decode from Base64
            roi_img = Image.open(BytesIO(base64.b64decode(roi_b64))).convert("RGB")
            periphery_img = Image.open(BytesIO(base64.b64decode(periphery_b64))).convert("RGB")
            
            # Standard transforms for image models
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
            ])

            roi_tensor = transform(roi_img) # Shape: [3, 200, 200]
            periphery_tensor = transform(periphery_img) # Shape: [3, H, W]

            # Resize periphery to match ROI height for tiling
            periphery_resized = T.functional.resize(periphery_tensor, size=[200, 200])

            # Tile images side-by-side [C, H, W] -> [C, H, 2*W]
            combined_tensor = torch.cat((roi_tensor, periphery_resized), dim=2) # dim=2 is width
            
            # Add a batch dimension for the model
            return combined_tensor.unsqueeze(0).to(self.device)

        except Exception as e:
            self.log(f"[ERROR] Failed to preprocess vision data: {e}")
            return None
    
    def _stop_components(self):
        """Stop all running components."""
        for comp_name, process in self.component_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                self.log(f"✅ Stopped component: {comp_name}")
            except Exception as e:
                self.log(f"[ERROR] Failed to stop component {comp_name}: {e}")
        self.component_processes.clear()

    @torch.no_grad()
    def _find_optimal_depth(self, d_lstm_input, head_name):
        """
        Processes an input through a D-LSTM head at various depths to find
        the shallowest depth where the output stabilizes. This version dynamically
        adjusts its search based on the model's physical max_depth.
        """
        head_network = self.pipeline.output_heads[head_name]
        initial_states = self.pipeline_states['heads'][head_name]
        
        # Get the physical max depth from the model itself
        physical_max_depth = head_network.max_depth
        
        # Get the current logical depth cap from the regulator
        logical_depth_cap = self.regulator.current_max_depth_cap
        
        # The search is limited by the smaller of the two
        search_limit = min(physical_max_depth, logical_depth_cap)

        # Dynamically create the search levels up to the current search limit
        base_levels = [4, 8, 16, 32, 64, 128, 256]
        depth_search_levels = [d for d in base_levels if d <= search_limit]
        
        # Ensure the final search_limit is always in the list
        if not depth_search_levels or depth_search_levels[-1] < search_limit:
            depth_search_levels.append(search_limit)

        similarity_threshold = 0.1
        last_output = None
        
        # If there's only one level to check (e.g., max_depth is small), just use it
        if len(depth_search_levels) == 1:
            return depth_search_levels[0]

        for i, depth in enumerate(depth_search_levels):
            # We run the forward pass up to the current search depth
            h_list, _ = head_network(
                d_lstm_input,
                initial_states['h'],
                initial_states['c'],
                depth
            )
            # The output is the hidden state of the final layer at that depth
            current_output = h_list[depth - 1]

            if i > 0:
                # Calculate the distance (change) from the previous depth's output
                distance = torch.linalg.norm(current_output - last_output)
                if distance < similarity_threshold:
                    # Output has stabilized, return the previous (shallower) depth
                    self.log(f"Depth found for '{head_name}': {depth_search_levels[i-1]} (stabilized at depth {depth})")
                    return depth_search_levels[i-1]
            
            last_output = current_output
        
        # If the output never stabilized, use the max allowed depth
        self.log(f"Depth for '{head_name}': {physical_max_depth} (did not stabilize)")
        return physical_max_depth

    def _load_master_hashes(self):
        """Loads the persistent master hash data from a JSON file."""
        filepath = os.path.join('models', 'master_hashes.json')
        self.log(f"DEBUG: Looking for master hashes at: {filepath}")
        self.log(f"DEBUG: File exists: {os.path.exists(filepath)}")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    master_hashes_data = json.load(f)
                self.log(f"DEBUG: Loaded data from file. Keys: {list(master_hashes_data.keys())}")
                if master_hashes_data:
                    self.hasher.set_master_hashes(master_hashes_data)
                    self.log(f"✅ Loaded {len(master_hashes_data)} master hashes.")
                    return True
            except Exception as e:
                self.log(f"[ERROR] Failed to load master hashes: {e}")
                import traceback
                self.log(f"Load error traceback: {traceback.format_exc()}")
        self.log("No master hashes found. Ready to learn.")
        return False

    def _load_pipeline_weights(self):
        """Loads the model's state dictionary from a file if it exists."""
        filepath = 'models/pipeline_weights.pth'
        if os.path.exists(filepath):
            try:
                # Load the state_dict, ensuring it's loaded to the correct device
                state_dict = torch.load(filepath, map_location=self.device)
                self.pipeline.load_state_dict(state_dict)
                self.log(f"🧠 Loaded model weights from {filepath}")
            except Exception as e:
                self.log(f"[ERROR] Failed to load model weights: {e}")
        else:
            self.log("No saved model weights found. Initializing with random weights.")

    def _save_master_hashes(self):
        """Saves the current master hash data to the JSON checkpoint file."""
        self.log(f"DEBUG: _save_master_hashes called. master_hash_data keys: {list(self.hasher.master_hash_data.keys())}")
        
        if not self.hasher.master_hash_data:
            self.log("No master hashes to save.")
            return

        # Prepare data for JSON serialization (convert numpy types if any)
        # This is a good practice, similar to the web_server.py helper
        def convert_to_native(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_to_native(i) for i in obj]
            return obj

        try:
            # Create a clean, serializable copy of the data
            data_to_save = convert_to_native(self.hasher.master_hash_data)
            self.log(f"DEBUG: Converted data. Keys to save: {list(data_to_save.keys())}")
            
            filepath = 'models/master_hashes.json'
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            
            self.log(f"💾 Saved {len(data_to_save)} master hashes to {filepath}")

        except Exception as e:
            self.log(f"[ERROR] Failed to save master hashes: {e}")
            import traceback
            self.log(f"Save error traceback: {traceback.format_exc()}")

    def _save_pipeline_weights(self):
        """Saves the state dictionary of the main PyTorch model pipeline."""
        try:
            filepath = 'models/pipeline_weights.pth'
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # We save the model's state_dict, which contains all weights and biases
            torch.save(self.pipeline.state_dict(), filepath)
            self.log(f"🧠 Saved model weights to {filepath}")
        except Exception as e:
            self.log(f"[ERROR] Failed to save model weights: {e}")

    def stop(self):
        """Signals the engine to stop its processing loop and saves checkpoints."""
        self.log("🛑 Stop command received. Shutting down engine loop...")
        
        # Stop all components first
        self._stop_components()
        
        # Finalize and save session data before stopping
        if self.is_running: # Only save if it was actually running
            self.tracker.stop_session(self.tick_count)

        # Save the tokenizer's learned knowledge.
        self.tokenizer.save_checkpoint()
        
        # Save the hasher's memory of master hashes.
        self.log(f"DEBUG: About to save master hashes. Count: {len(self.hasher.master_hash_data)}")
        self._save_master_hashes()
        
        # --- ADD THIS LINE ---
        self._save_pipeline_weights()

        self.is_running = False

    def run(self):
        """The main loop for the engine thread, now implemented as a state machine."""
        self.is_running = True
        self.start_time = time.time()
        self.tick_count = 0
        self.ticks_spent_awake = 0
        self.agent_state = "AWAKE"
        
        self.tracker.start_session()
        self.log("Engine processing thread has started.")
        self.log(f"Initial state: {self.agent_state}")

        target_interval = 1.0 / 20.0
        last_tick_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - last_tick_time < target_interval:
                    time.sleep(0.001) # Prevent busy waiting
                    continue

                last_tick_time = current_time

                # --- STATE MACHINE LOGIC ---
                # 1. Check with the Regulator to determine the correct state
                current_energy = self.get_energy()
                current_boredom = self.get_boredom()
                self.agent_state = self.regulator.check_state(
                    current_energy, current_boredom, self.ticks_spent_awake
                )

                # 2. Execute the logic for the current state
                if self.agent_state == "AWAKE" or self.agent_state == "CRYING":
                    # For now, CRYING and AWAKE states both run the main tick cycle.
                    # We will add the special 'rut-breaking' logic for CRYING later.
                    self._tick_cycle()
                
                elif self.agent_state == "SLEEPING":
                    # --- SLEEP CYCLE PLACEHOLDER ---
                    # This is where the training logic will go.
                    # For now, we simulate sleep and then wake up.
                    self.log(f"Agent state is now SLEEPING. Pausing for {self.regulator.sleep_duration_ticks / 20.0:.1f}s...")
                    
                    # Calculate average novelty from the last awake cycle
                    # This is a simple placeholder - in a full implementation, this would
                    # be calculated from stored experiences during the awake period
                    avg_novelty_scores = list(self.head_novelty_scores.values())
                    novelty_average = sum(avg_novelty_scores) / len(avg_novelty_scores) if avg_novelty_scores else 0.5
                    
                    # Update growth stage based on novelty experienced
                    self.regulator.update_growth_stage(novelty_average)
                    
                    # TODO: Implement the _sleep_cycle() method with ReplayBuffer and training.
                    time.sleep(self.regulator.sleep_duration_ticks * target_interval) # Use regulator's sleep duration
                    
                    self.log("...waking up.")
                    self.agent_state = "AWAKE"
                    self.ticks_spent_awake = 0 # Reset the awake counter
            
            except Exception as e:
                self.log(f"CRITICAL ERROR in main run loop: {str(e)}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                self.is_running = False
        
        self.log("Engine loop has terminated.")

    def _tick_cycle(self):
        """Performs one cycle of the internal engine."""
        try:
            start_time = time.time()
            self.tick_count += 1
            self.ticks_spent_awake += 1

            # --- VISION NOVELTY ENGINE ---
            vision_data = self.sensory_info.get('vision', {})
            visual_novelty_vector = np.zeros(self.vae_encoder.latent_dim) # Default to zero vector

            combined_image_tensor = self._preprocess_vision_data(vision_data)
            
            if combined_image_tensor is not None:
                # Pass the combined image through the frozen VAE encoder
                current_mu, _ = self.vae_encoder(combined_image_tensor)
                
                # Calculate the difference vector if we have a previous vector
                if self.previous_latent_vector is not None:
                    diff_tensor = current_mu - self.previous_latent_vector
                    visual_novelty_vector = diff_tensor.cpu().numpy().flatten()
                
                # Update the memory for the next tick
                self.previous_latent_vector = current_mu
            
            # Add the final vectors to sensory_info for the tokenizer to use
            if 'vision' not in self.sensory_info: self.sensory_info['vision'] = {}
            self.sensory_info['vision']['visual_novelty_vector'] = visual_novelty_vector
            
            # --- ADD THIS ---
            # Also add the static latent vector for the current frame
            if self.previous_latent_vector is not None:
                # We use previous_latent_vector as it's the one that produced the novelty signal for this tick
                static_latent_vector = self.previous_latent_vector.cpu().numpy().flatten()
                self.sensory_info['vision']['static_latent_vector'] = static_latent_vector

            # --- 1. Real Sensory Data & Tokenization ---
            # Use the real sensory data collected from components
            # If no real data is available, fall back to simulated data
            if not self.sensory_info:
                # Fallback to simulated data if no components are running
                self.sensory_info = {
                    'mouse': {
                        'x': np.random.randint(0, 1920),
                        'y': np.random.randint(0, 1080)
                    }
                }
            
            # Use the tokenizer to process the real sensory data
            token_vector_np = self.tokenizer.tokenize(self.sensory_info)

            # --- VECTOR AUTOPSY: Diagnostic logging every 20 ticks ---
            if self.tick_count % 20 == 0:
                self.log("=== VECTOR AUTOPSY START ===")
                self.log(f"Token Vector Shape: {token_vector_np.shape}")
                self.log(f"Token Vector dtype: {token_vector_np.dtype}")
                
                # Overall statistics
                self.log(f"Min: {np.min(token_vector_np):.6f}")
                self.log(f"Max: {np.max(token_vector_np):.6f}")
                self.log(f"Mean: {np.mean(token_vector_np):.6f}")
                self.log(f"Std: {np.std(token_vector_np):.6f}")
                self.log(f"Variance: {np.var(token_vector_np):.6f}")
                
                # Check for zero/constant regions
                zero_count = np.sum(token_vector_np == 0.0)
                self.log(f"Zero elements: {zero_count}/{len(token_vector_np)} ({100*zero_count/len(token_vector_np):.1f}%)")
                
                # Check for NaN/inf values
                nan_count = np.sum(np.isnan(token_vector_np))
                inf_count = np.sum(np.isinf(token_vector_np))
                self.log(f"NaN count: {nan_count}, Inf count: {inf_count}")
                
                # Histogram of values to see distribution
                unique_values = len(np.unique(token_vector_np))
                self.log(f"Unique values: {unique_values}/{len(token_vector_np)} ({100*unique_values/len(token_vector_np):.1f}%)")
                
                # Show first 20 values for manual inspection
                sample_values = token_vector_np[:20] if len(token_vector_np) >= 20 else token_vector_np
                self.log(f"First 20 values: {sample_values}")
                
                # Check what sensory data we actually received
                self.log(f"Sensory components available: {list(self.sensory_info.keys())}")
                for comp_name, comp_data in self.sensory_info.items():
                    if isinstance(comp_data, dict):
                        self.log(f"  {comp_name}: {list(comp_data.keys())}")
                    else:
                        self.log(f"  {comp_name}: {type(comp_data)}")
                
                self.log("=== VECTOR AUTOPSY END ===")

            # --- NEW: Hashing Step ---
            # Use the LSH system to hash the token vector
            hash_result = self.hasher.query_and_update(token_vector_np, self.tick_count)
            
            # Convert to a tensor for the model
            input_tensor = torch.from_numpy(token_vector_np).to(self.device)

            # --- 2. Model Processing with Action Selection ---
            with torch.no_grad(): # <-- ADD THIS LINE and indent the entire block below
                # --- A: Credit Assignment (based on the PREVIOUS tick's action) ---
                current_novelty_score = 1.0 if hash_result['status'] == 'novel' else 0.0
                if len(self.action_outcome_buffer) > 0:
                    last_tick_data = self.action_outcome_buffer[-1]
                    # If this is the tick right after the action was taken...
                    if last_tick_data['tick'] == self.tick_count - 1:
                        novelty_delta = current_novelty_score - last_tick_data['novelty_before']
                        if novelty_delta > 0:
                            # Reward the heads that were active
                            for head_name in last_tick_data['active_heads']:
                                score = self.head_novelty_scores.get(head_name, 1.0)
                                # Update score using a moving average
                                self.head_novelty_scores[head_name] = (score * 0.95) + (novelty_delta * 0.05)
                
                # Decay all scores slightly so they don't grow forever
                for head_name in self.head_novelty_scores:
                    self.head_novelty_scores[head_name] *= 0.999

                # --- B: Action Selection (for the CURRENT tick) ---
                active_heads = self.action_selector.select_actions(self, self.head_novelty_scores)
                self.log(f"Active Heads ({len(active_heads)}): {', '.join(active_heads)}")
                
                d_lstm_input = None # This will hold the input for the D-LSTM heads
                
                # --- Trunk LSTM Processing (or Cache Retrieval) ---
                if hash_result['status'] == 'master' and 'trunk_output' in self.hasher.master_hash_data.get(hash_result['hash'], {}):
                    # CACHE HIT: Master hash found with a cached output
                    self.log("CACHE HIT (Trunk)")
                    cached_output_list = self.hasher.master_hash_data[hash_result['hash']]['trunk_output']
                    d_lstm_input = torch.tensor(cached_output_list, device=self.device).view(1, 1, -1)
                else:
                    # CACHE MISS: This is the "first pass" for this state
                    self.log("CACHE MISS (Trunk)")
                    x_for_trunk = input_tensor.view(1, 1, -1)
                    p_out, p_states = self.pipeline.pattern_lstm(x_for_trunk, self.pipeline_states['pattern'])
                    c_out, c_states = self.pipeline.compression_lstm(p_out, self.pipeline_states['compression'])
                    
                    self.pipeline_states['pattern'] = p_states
                    self.pipeline_states['compression'] = c_states
                    d_lstm_input = c_out
                    
                    if hash_result['status'] == 'master':
                        output_list = d_lstm_input.cpu().detach().numpy().flatten().tolist()
                        self.hasher.master_hash_data[hash_result['hash']]['trunk_output'] = output_list

                # --- Optimal Depth Search (or Retrieval) ---
                required_depths = {}
                master_data = self.hasher.master_hash_data.get(hash_result['hash'])

                if hash_result['status'] == 'master' and master_data and master_data.get('optimal_depths'):
                    # DEPTHS FOUND: Retrieve saved depths for this master hash
                    required_depths = master_data['optimal_depths']
                else:
                    # DEPTHS NOT FOUND: This is the "first pass" for the heads
                    self.log(f"Finding optimal depths for hash {hash_result['hash'][:8]}...")
                    for head_name in self.pipeline.output_heads.keys():
                        depth = self._find_optimal_depth(d_lstm_input, head_name)
                        required_depths[head_name] = depth
                    
                    # Save the newly found depths to the master hash data
                    if hash_result['status'] == 'master':
                        self.hasher.master_hash_data[hash_result['hash']]['optimal_depths'] = required_depths

                # --- C: Head Processing (now uses the active_heads list) ---
                final_outputs = {}
                # ONLY loop through the heads selected by the ActionSelector
                for head_name in active_heads:
                    if head_name in self.pipeline.output_heads:
                        head_network = self.pipeline.output_heads[head_name]
                        # The depth still comes from the hasher's memory
                        depth = required_depths.get(head_name, 32) # Default to 32 if not found
                        
                        # Run the forward pass for this specific head
                        h_list, c_list = head_network(
                            d_lstm_input, # Use the output from the cache or trunk
                            self.pipeline_states['heads'][head_name]['h'],
                            self.pipeline_states['heads'][head_name]['c'],
                            depth
                        )
                        
                        # Update the states for this head for the next tick
                        self.pipeline_states['heads'][head_name] = {'h': h_list, 'c': c_list}
                        
                        # Get the final feature vector from the correct depth
                        final_feature_vector = h_list[depth - 1].squeeze(0).squeeze(0)
                        
                        # Pass it through the final output layer if it exists
                        if head_name in self.pipeline.output_layers:
                            final_output_tensor = self.pipeline.output_layers[head_name](final_feature_vector)
                        else:
                            final_output_tensor = final_feature_vector
                        
                        final_outputs[head_name] = final_output_tensor

                # --- D: Store Action for Next Tick's Credit Assignment ---
                self.action_outcome_buffer.append({
                    'tick': self.tick_count,
                    'active_heads': active_heads,
                    'novelty_before': current_novelty_score
                })
            
            # CRITICAL: We no longer need the monolithic call to self.pipeline(...)
            # The logic above has replaced it.

            # --- 3. Calculate TPS (Ticks Per Second) ---
            current_time = time.time()
            self.tps_window_ticks += 1
            
            # Update TPS every second using a sliding window
            if current_time - self.tps_window_start >= 1.0:
                window_duration = current_time - self.tps_window_start
                self.tps = self.tps_window_ticks / window_duration
                self.last_tps_update = current_time
                
                # Debug: Log TPS calculation
                self.log(f"TPS Update: {self.tps_window_ticks} ticks in {window_duration:.2f}s = {self.tps:.1f} TPS")
                
                # Reset the window for next calculation
                self.tps_window_start = current_time
                self.tps_window_ticks = 0
            elif self.tick_count <= 20:
                # For the first 20 ticks, estimate TPS based on current rate
                elapsed_since_start = current_time - self.start_time
                if elapsed_since_start > 0:
                    self.tps = self.tick_count / elapsed_since_start

            # --- 4. Decode Outputs and Send Updates to GUI ---
            
            # A. Decode all active head outputs for the monitor
            decoded_outputs = {}
            with torch.no_grad():
                for head_name, output_tensor in final_outputs.items():
                    decoded_string = self.tokenizer.decode(head_name, output_tensor)
                    decoded_outputs[head_name] = decoded_string

            # B. Determine agent's high-level status
            current_energy = self.get_energy()
            current_boredom = self.get_boredom()
            agent_status = "Awake"
            if current_energy < 0.2:
                agent_status = "Asleep"
            elif current_boredom > 0.9:
                agent_status = "Crying"

            # C. Assemble the full status package for the GUI
            vision_data = self.sensory_info.get('vision', {})
            status_update = {
                'type': 'status',
                'session_tick': self.tick_count,
                'active_tokens': self.tokenizer.get_active_token_count(),
                'tps': round(self.tps, 1),
                'lifetime_stats': {
                    'total_ticks': self.tracker.stats['total_ticks_ever'] + self.tick_count,
                    'total_runtime_hr': round( (self.tracker.stats['total_runtime_seconds'] + (time.time() - (self.tracker.current_session_start_time or time.time()))) / 3600, 2),
                    'session_count': self.tracker.stats['session_count']
                },
                'is_running': self.is_running,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'active_heads': active_heads,
                'energy': round(current_energy, 2),
                'boredom': round(current_boredom, 2),
                'novelty_scores': {k: round(v, 3) for k, v in self.head_novelty_scores.items()},
                'agent_status': agent_status,
                'decoded_outputs': decoded_outputs, # NEW: Add the full decoded output dictionary
                
                # --- MODIFIED LINES ---
                'display_image_b64': vision_data.get('display_image_b64'),
                'roi_image_b64': vision_data.get('roi_image_b64')
            }
            # Debug: Log what we're sending
            if self.tick_count % 20 == 0:  # Log every 20th tick to avoid spam
                self.log(f"Status Update - Tick: {self.tick_count}, TPS: {self.tps:.1f}")
            
            # Clear vision data after it's been packaged
            if 'vision' in self.sensory_info:
                 self.sensory_info['vision'] = {}
            
            # D. Put the update in the queue using a non-blocking, "UDP-style" approach.
            try:
                # Try to add the item without waiting.
                self.update_queue.put_nowait(status_update)
            except queue.Full:
                # If the queue is full, the GUI is lagging.
                # Discard the oldest update to make room for the new one.
                try:
                    self.update_queue.get_nowait()
                    self.update_queue.put_nowait(status_update)
                except queue.Empty:
                    # This is a rare race condition, but we handle it just in case.
                    pass

            # --- NEW: Send Hash Memory Snapshot Periodically ---
            if self.tick_count % 20 == 0: # Send every 20 ticks
                memory_snapshot = {
                    'master_hashes': self.hasher.master_hash_data,
                    'novel_hashes': self.hasher.novel_hash_cache
                }
                # This is the message the hashes.js is waiting for
                try:
                    self.update_queue.put_nowait({'type': 'hash_memory_update', 'data': memory_snapshot})
                except queue.Full:
                    try:
                        self.update_queue.get_nowait()
                        self.update_queue.put_nowait({'type': 'hash_memory_update', 'data': memory_snapshot})
                    except queue.Empty:
                        pass
            
        except Exception as e:
            # Log any errors that occur during tick processing
            self.log(f"ERROR in tick cycle: {str(e)}")
            self.log(f"Error type: {type(e).__name__}")
            import traceback
            self.log(f"Tick cycle traceback: {traceback.format_exc()}")
            # Still send a basic status update even if there's an error
            status_update = {
                'type': 'status',
                'session_tick': self.tick_count,
                'active_tokens': self.tokenizer.get_active_token_count(),
                'tps': round(self.tps, 1),
                'lifetime_stats': {
                    'total_ticks': self.tracker.stats['total_ticks_ever'] + self.tick_count,
                    'total_runtime_hr': round( (self.tracker.stats['total_runtime_seconds'] + (time.time() - (self.tracker.current_session_start_time or time.time()))) / 3600, 2),
                    'session_count': self.tracker.stats['session_count']
                },
                'is_running': self.is_running,
                'processing_time_ms': 0,
                'model_outputs': {'error': str(e)},
                'error': True,
                'active_heads': ['internal_thought'],  # Fallback to just internal thought on error
                'energy': round(self.get_energy(), 2),
                'boredom': round(self.get_boredom(), 2),
                'novelty_scores': {k: round(v, 2) for k, v in self.head_novelty_scores.items()}
            }
            
            # Put the error status update in the queue using non-blocking approach
            try:
                self.update_queue.put_nowait(status_update)
            except queue.Full:
                try:
                    self.update_queue.get_nowait()
                    self.update_queue.put_nowait(status_update)
                except queue.Empty:
                    pass

            # Still send hash memory data even on error (if available)
            if hasattr(self, 'hasher'):
                memory_snapshot = {
                    'master_hashes': self.hasher.master_hash_data,
                    'novel_hashes': self.hasher.novel_hash_cache
                }
                try:
                    self.update_queue.put_nowait({'type': 'hash_memory_update', 'data': memory_snapshot})
                except queue.Full:
                    try:
                        self.update_queue.get_nowait()
                        self.update_queue.put_nowait({'type': 'hash_memory_update', 'data': memory_snapshot})
                    except queue.Empty:
                        pass 