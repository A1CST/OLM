import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    A circular buffer for storing and sampling experiences during the OLM's awake cycle.
    Used during the dreaming phase to replay memories for training.
    """
    
    def __init__(self, max_size=10000):
        """
        Initialize the replay buffer.
        
        Args:
            max_size (int): Maximum number of experiences to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add_experience(self, token_vector, hash_result, required_depths, active_heads=None, novelty_scores=None):
        """
        Add a new experience to the buffer.
        
        Args:
            token_vector (np.ndarray): The tokenized sensory input
            hash_result (dict): The novelty hash result
            required_depths (dict): The processing depths for each head
            active_heads (list): Which heads were active this tick
            novelty_scores (dict): Novelty scores for each head
        """
        experience = {
            'token_vector': token_vector.copy() if isinstance(token_vector, np.ndarray) else np.array(token_vector),
            'hash_result': hash_result.copy() if hash_result else {},
            'required_depths': required_depths.copy() if required_depths else {},
            'active_heads': active_heads.copy() if active_heads else [],
            'novelty_scores': novelty_scores.copy() if novelty_scores else {}
        }
        
        self.buffer.append(experience)
    
    def sample(self, batch_size=1):
        """
        Sample a random batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list: List of experience dictionaries, or empty list if buffer is empty
        """
        if self.is_empty():
            return []
        
        # Don't sample more than we have
        actual_batch_size = min(batch_size, len(self.buffer))
        
        return random.sample(list(self.buffer), actual_batch_size)
    
    def sample_recent(self, batch_size=1, recent_fraction=0.2):
        """
        Sample from the most recent experiences (for more relevant dreams).
        
        Args:
            batch_size (int): Number of experiences to sample
            recent_fraction (float): Fraction of buffer to consider "recent"
            
        Returns:
            list: List of recent experience dictionaries
        """
        if self.is_empty():
            return []
        
        recent_count = max(1, int(len(self.buffer) * recent_fraction))
        recent_experiences = list(self.buffer)[-recent_count:]
        
        actual_batch_size = min(batch_size, len(recent_experiences))
        
        return random.sample(recent_experiences, actual_batch_size)
    
    def sample_novel(self, batch_size=1):
        """
        Sample experiences that were marked as novel (for focused learning).
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list: List of novel experience dictionaries
        """
        if self.is_empty():
            return []
        
        # Filter for novel experiences
        novel_experiences = [
            exp for exp in self.buffer 
            if exp.get('hash_result', {}).get('status') == 'novel'
        ]
        
        if not novel_experiences:
            # Fall back to regular sampling if no novel experiences
            return self.sample(batch_size)
        
        actual_batch_size = min(batch_size, len(novel_experiences))
        
        return random.sample(novel_experiences, actual_batch_size)
    
    def is_empty(self):
        """Check if the buffer is empty."""
        return len(self.buffer) == 0
    
    def size(self):
        """Get the current number of experiences in the buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences from the buffer."""
        self.buffer.clear()
    
    def get_stats(self):
        """Get statistics about the buffer contents."""
        if self.is_empty():
            return {
                'size': 0,
                'novel_count': 0,
                'familiar_count': 0,
                'avg_novelty_score': 0.0
            }
        
        novel_count = sum(1 for exp in self.buffer if exp.get('hash_result', {}).get('status') == 'novel')
        familiar_count = len(self.buffer) - novel_count
        
        # Calculate average novelty across all heads
        all_novelty_scores = []
        for exp in self.buffer:
            novelty_scores = exp.get('novelty_scores', {})
            if novelty_scores:
                all_novelty_scores.extend(novelty_scores.values())
        
        avg_novelty = np.mean(all_novelty_scores) if all_novelty_scores else 0.0
        
        return {
            'size': len(self.buffer),
            'novel_count': novel_count,
            'familiar_count': familiar_count,
            'avg_novelty_score': float(avg_novelty)
        }