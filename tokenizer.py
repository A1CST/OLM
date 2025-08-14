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

    def decode(self, head_name, output_vector):
        """Decodes an output vector into a human-readable format."""
        if not isinstance(output_vector, np.ndarray):
            output_vector = output_vector.cpu().detach().numpy()
        output_vector = output_vector.flatten()

        if head_name == "speech":
            if not hasattr(self, 'index_char_map'):
                self.index_char_map = {i: char for char, i in self.char_map.items()}
            top_index = np.argmax(output_vector)
            return self.index_char_map.get(top_index, '?')
        elif head_name == "action":
            top_index = np.argmax(output_vector)
            return self.action_map.get(top_index, 'UNKNOWN_ACTION')
        elif head_name in ["video", "image", "internal_thought"]:
            vec_preview = ", ".join([f"{x:.2f}" for x in output_vector[:4]])
            return f"Vector({head_name}): [{vec_preview}, ...]"
        else:
            return f"Unknown head type: {head_name}" 