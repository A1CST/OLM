import numpy as np
import json
import os

class SensoryTokenizer:
    """
    Converts raw sensory_info into a normalized, COMPACT vector and decodes model outputs.
    This version is stateful: it learns the min/max ranges of sensory data
    and can save/load its state from a checkpoint.
    """
    # NEW VECTOR SIZE: 10(mouse) + 4(internals) + 64(vision_novelty) + 64(static_latent) = 142
    def __init__(self, vector_size=142, checkpoint_path='models/tokenizer_checkpoint.json'):
        self.vector_size = vector_size
        self.checkpoint_path = checkpoint_path
        self.normalization_stats = {}
        self.char_map = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\"-")}
        self.action_map = {
            0: "MOVE_MOUSE_REL(dx,dy)",
            1: "CLICK_MOUSE(button)",
            2: "TYPE_TEXT(text)",
            3: "NONE"
        }
        self.load_checkpoint()

    def save_checkpoint(self):
        """Saves the learned normalization stats to a file."""
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            data_to_save = {'normalization_stats': self.normalization_stats}
            with open(self.checkpoint_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Failed to save tokenizer checkpoint: {e}")

    def load_checkpoint(self):
        """Loads normalization stats from a file if it exists."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                self.normalization_stats = checkpoint_data.get('normalization_stats', {})
                print(f"Tokenizer checkpoint loaded from {self.checkpoint_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load tokenizer checkpoint: {e}")
        else:
            print("No tokenizer checkpoint found. Initializing with default values.")

    def update_and_normalize(self, feature_name, value):
        """Updates the min/max stats for a feature and returns its normalized value."""
        if feature_name not in self.normalization_stats:
            self.normalization_stats[feature_name] = {'min': float(value), 'max': float(value)}
        
        stats = self.normalization_stats[feature_name]
        stats['min'] = min(stats['min'], float(value))
        stats['max'] = max(stats['max'], float(value))
        
        denominator = stats['max'] - stats['min']
        return (float(value) - stats['min']) / denominator if denominator > 0 else 0.0

    def tokenize(self, sensory_info):
        """Takes sensory_info, updates stats, and returns a normalized vector."""
        token_vector = np.zeros(self.vector_size, dtype=np.float32)
        
        # --- Mouse Features (Slots 0-9) ---
        mouse_data = sensory_info.get('mouse', {})
        token_vector[0] = self.update_and_normalize('mouse_x', mouse_data.get('x', 0))
        token_vector[1] = self.update_and_normalize('mouse_y', mouse_data.get('y', 0))
        token_vector[2] = self.update_and_normalize('mouse_delta', mouse_data.get('delta', 0))
        token_vector[3] = self.update_and_normalize('mouse_scroll', mouse_data.get('scroll', 0))
        token_vector[4] = mouse_data.get('left_single_click', 0.0)
        token_vector[5] = mouse_data.get('right_single_click', 0.0)
        token_vector[6] = mouse_data.get('left_hold', 0.0)
        
        # --- Internals Features (Slots 10-13) ---
        internals_data = sensory_info.get('internals', {})
        token_vector[10] = self.update_and_normalize('cpu_percent', internals_data.get('cpu_percent', 0))
        token_vector[11] = self.update_and_normalize('memory_percent', internals_data.get('memory_percent', 0))
        token_vector[12] = self.update_and_normalize('gpu_load', internals_data.get('gpu_load', 0))
        token_vector[13] = self.update_and_normalize('temperature_c', internals_data.get('temperature_c', 0))

        # --- Vision Novelty Features (Slots 14-77) ---
        vision_data = sensory_info.get('vision', {})
        novelty_vector = vision_data.get('visual_novelty_vector', np.zeros(64))
        clipped_vector = np.clip(novelty_vector, -1.0, 1.0)
        token_vector[14:78] = clipped_vector

        # --- Static Latent Vector Features (Slots 78-141) ---
        static_vector = vision_data.get('static_latent_vector', np.zeros(64))
        for i, value in enumerate(static_vector):
            feature_name = f'static_latent_{i}'
            token_vector[78 + i] = self.update_and_normalize(feature_name, value)

        return token_vector

    def get_active_token_count(self):
        """Returns the number of features the tokenizer has learned to track."""
        return len(self.normalization_stats)

    def decode(self, head_name, output_vector, temperature=0.0):
        """
        Decodes an output vector into a human-readable format.
        
        Args:
            head_name (str): Name of the output head
            output_vector: The output tensor/array from the head
            temperature (float): Temperature for sampling. 0.0 = deterministic (argmax), 
                               higher values = more random sampling
        """
        if not isinstance(output_vector, np.ndarray):
            # Handle different input types
            if hasattr(output_vector, 'cpu'):
                # PyTorch tensor
                output_vector = output_vector.cpu().detach().numpy()
            elif isinstance(output_vector, list):
                # Convert list to numpy array
                output_vector = np.array(output_vector)
            else:
                # Try to convert to numpy array
                output_vector = np.array(output_vector)
        output_vector = output_vector.flatten()

        if head_name == "speech":
            if not hasattr(self, 'index_char_map'):
                self.index_char_map = {i: char for char, i in self.char_map.items()}
            
            if temperature > 0:
                # Apply temperature to logits
                logits = output_vector / temperature
                # Calculate probabilities using softmax
                exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
                probabilities = exp_logits / np.sum(exp_logits)
                # Sample from the distribution
                top_index = np.random.choice(len(probabilities), p=probabilities)
            else:
                # Temperature of 0 means deterministic argmax
                top_index = np.argmax(output_vector)
            
            return self.index_char_map.get(top_index, '?')
            
        elif head_name == "action":
            if temperature > 0:
                logits = output_vector / temperature
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)
                top_index = np.random.choice(len(probabilities), p=probabilities)
            else:
                top_index = np.argmax(output_vector)
            
            return self.action_map.get(top_index, 'UNKNOWN_ACTION')
            
        elif head_name in ["video", "image", "internal_thought"]:
            # For vector outputs, temperature doesn't apply - just show preview
            vec_preview = ", ".join([f"{x:.2f}" for x in output_vector[:4]])
            temp_suffix = f" (T={temperature:.1f})" if temperature > 0 else ""
            return f"Vector({head_name}): [{vec_preview}, ...]{temp_suffix}"
        else:
            return f"Unknown head type: {head_name}"

    def detokenize_internal_thought(self, thought_vector):
        """
        Converts internal thought vector back to sensory data format for feedback loop.
        
        Maps the 64-dimensional internal thought output back to the 142D sensory vector format.
        This creates a feedback mechanism where internal thoughts influence future processing.
        
        Args:
            thought_vector: numpy array or torch tensor of internal thought output (64D)
            
        Returns:
            dict: sensory_info format dictionary with internal thought data
        """
        if not isinstance(thought_vector, np.ndarray):
            thought_vector = thought_vector.cpu().detach().numpy()
        thought_vector = thought_vector.flatten()
        
        # Ensure we have the expected 64-dimensional vector
        if len(thought_vector) != 64:
            # Pad or truncate to 64 dimensions
            if len(thought_vector) < 64:
                padded = np.zeros(64)
                padded[:len(thought_vector)] = thought_vector
                thought_vector = padded
            else:
                thought_vector = thought_vector[:64]
        
        # Create internal feedback data structure
        # We'll map portions of the internal thought to different sensory modalities
        # This creates a rich internal representation that can influence all aspects
        
        # Split the 64D thought vector into logical segments
        mouse_segment = thought_vector[0:8]        # 8 values for mouse influence
        internals_segment = thought_vector[8:12]   # 4 values for system awareness
        vision_novelty_segment = thought_vector[12:48]  # 36 values for visual influence
        vision_static_segment = thought_vector[48:64]   # 16 values for visual memory
        
        # Create sensory data structure with internal thought influence
        internal_sensory_data = {
            'mouse': {
                # Map internal thoughts to subtle mouse influences
                'x': float(mouse_segment[0] * 100),  # Small positional bias
                'y': float(mouse_segment[1] * 100),
                'delta': abs(float(mouse_segment[2] * 10)),  # Movement influence
                'scroll': float(mouse_segment[3] * 2),
                'left_single_click': max(0.0, float(mouse_segment[4])),
                'right_single_click': max(0.0, float(mouse_segment[5])),
                'left_hold': max(0.0, float(mouse_segment[6])),
                'internal_thought_influence': True  # Flag to identify this as internal
            },
            
            'internals': {
                # Map internal thoughts to system awareness
                'cpu_percent': abs(float(internals_segment[0] * 20)),
                'memory_percent': abs(float(internals_segment[1] * 20)),
                'gpu_load': abs(float(internals_segment[2] * 20)),
                'temperature_c': abs(float(internals_segment[3] * 10)),
                'internal_thought_influence': True
            },
            
            'vision': {
                # Map internal thoughts to visual processing influence
                'visual_novelty_vector': np.concatenate([
                    vision_novelty_segment * 0.1,  # Use our 36 values  
                    np.zeros(28)  # Pad to 64 total for compatibility
                ]),
                'static_latent_vector': np.concatenate([
                    vision_static_segment * 0.1,  # Use our 16 values
                    np.zeros(48)  # Pad to 64 total for compatibility
                ]),
                'internal_thought_influence': True,
                'original_thought_vector': thought_vector  # Keep original for analysis
            },
            
            # Add metadata about this internal thought
            '_internal_thought_metadata': {
                'source': 'internal_thought_head',
                'vector_norm': float(np.linalg.norm(thought_vector)),
                'max_activation': float(np.max(np.abs(thought_vector))),
                'active_dimensions': int(np.sum(np.abs(thought_vector) > 0.01)),
                'timestamp': self._get_current_timestamp()
            }
        }
        
        return internal_sensory_data
    
    def _get_current_timestamp(self):
        """Helper to get current timestamp for internal thought tracking."""
        import time
        return time.time() 