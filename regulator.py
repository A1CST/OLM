import numpy as np

class Regulator:
    """
    Governs the agent's metabolic and learning cycles. It manages the agent's
    growth stage, dynamically adjusts its core parameters, and determines
    whether the agent should be AWAKE, SLEEPING, or CRYING.
    """
    def __init__(self, physical_max_depth=16):
        self.physical_max_depth = physical_max_depth
        self.growth_stage = 1.0 # Start at stage 1
        
        # Initialize all dynamic parameters
        self.energy_cap = 0.0
        self.awake_duration_ticks = 0
        self.sleep_duration_ticks = 0
        self.training_sample_size = 0
        self.current_max_depth_cap = 0
        self.is_crying = False
        
        # Trigger thresholds (hyperparameters to be tuned)
        self.sleep_energy_threshold = 0.2
        self.cry_boredom_threshold = 0.9
        
        # Update parameters based on the initial growth stage
        self._update_parameters()

    def _update_parameters(self):
        """
        Calculates all dynamic parameters based on the current growth_stage.
        This is the core of the agent's maturation process.
        """
        # Energy Cap: Starts low, increases with growth. (e.g., from 0.4 to 1.0)
        self.energy_cap = min(1.0, 0.4 + self.growth_stage * 0.05)
        
        # Awake Duration: Stays awake longer as it matures. (e.g., from ~1 min to 10+ mins)
        self.awake_duration_ticks = int(1200 + self.growth_stage * 500) # 20 TPS * 60s = 1200
        
        # Sleep Duration: Now dynamic based on dream count (calculated per sleep cycle)
        # This is no longer a fixed duration but depends on thoughts during awake cycle
        
        # Training Sample Size: Learns from more memories as it grows.
        self.training_sample_size = int(100 + self.growth_stage * 50)
        
        # Logical Depth Cap: Potential for "deeper thought" increases with growth.
        self.current_max_depth_cap = int(min(self.physical_max_depth, 4 + self.growth_stage))

    def update_growth_stage(self, novelty_average):
        """
        Updates the growth stage based on the novelty experienced during
        the last awake cycle. High novelty leads to more growth.
        """
        # The amount of growth is proportional to the novelty experienced.
        # A scaling factor determines how fast the agent "ages".
        growth_increment = novelty_average * 0.1 # Hyperparameter
        self.growth_stage += growth_increment
        
        # After growing, update all parameters for the next cycle
        self._update_parameters()
        print(f"[REGULATOR] Growth detected! New stage: {self.growth_stage:.2f}. Parameters updated.")

    def check_state(self, current_energy, current_boredom, ticks_spent_awake):
        """
        The main decision-making function. Determines the agent's next state.
        
        Returns:
            str: The state the agent should be in: "AWAKE", "SLEEPING", or "CRYING".
        """
        # Priority 1: Must sleep if out of energy or awake for too long.
        if current_energy < self.sleep_energy_threshold or ticks_spent_awake > self.awake_duration_ticks:
            self.is_crying = False # Can't cry if sleeping
            return "SLEEPING"
        
        # Priority 2: Enter "Crying" state if extremely bored and has energy.
        if current_boredom > self.cry_boredom_threshold:
            self.is_crying = True
            return "CRYING"
            
        # Default State: If none of the above, remain awake.
        self.is_crying = False
        return "AWAKE"

    def calculate_dream_count(self, thought_count):
        """
        Calculates the number of dreams to process during sleep,
        scaled by the agent's growth stage.
        
        Args:
            thought_count (int): Number of internal thoughts during the awake cycle
            
        Returns:
            int: Number of dream iterations to perform
        """
        # A more mature agent can process more thoughts per sleep cycle
        dream_factor = 0.5 + (self.growth_stage * 0.1)  # Hyperparameter
        dream_count = int(thought_count * dream_factor)+100
        
        # Ensure we have at least 1 dream and cap at reasonable maximum
        return max(1, min(dream_count, 100)) 