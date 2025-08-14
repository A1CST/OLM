import numpy as np

class ActionSelector:
    """
    A lightweight heuristic module that decides which D-LSTM heads to run
    based on predicted novelty, boredom, and energy constraints.
    """
    def __init__(self, head_configs):
        # Calculate a base energy cost for running each head,
        # proportional to its complexity (hidden_size).
        self.head_energy_costs = {}
        cost_per_neuron = 0.0001 # Hyperparameter to tune
        for name, config in head_configs.items():
            self.head_energy_costs[name] = config['hidden_size'] * cost_per_neuron
        
        # Override specific head costs for better balance
        self.head_energy_costs['internal_thought'] = 0.03  # Much lower cost for internal thoughts

    def select_actions(self, olm_agent, head_novelty_scores):
        """
        Selects which D-LSTM heads to activate for the current tick.

        Returns:
            list: A list of head names to activate.
        """
        # Rule #1: The 'internal_thought' head is always active.
        active_heads = ['internal_thought']

        # Get the agent's current state from the OLM
        current_energy = olm_agent.get_energy()
        current_boredom = olm_agent.get_boredom() # This is our "motivation" signal

        # Define a processing budget for this tick (e.g., 5% of current energy)
        processing_budget = current_energy * 0.05
        
        # Subtract the cost of the mandatory 'internal_thought' head
        processing_budget -= self.head_energy_costs.get('internal_thought', 0.1)

        # Get a list of other heads to consider, excluding 'internal_thought'
        candidate_heads = {
            name: score for name, score in head_novelty_scores.items()
            if name != 'internal_thought'
        }

        # Calculate a priority score for each candidate
        priority_scores = {
            name: score * current_boredom # Higher boredom amplifies the desire to act
            for name, score in candidate_heads.items()
        }

        # Sort candidates by their priority score, highest first
        sorted_candidates = sorted(
            priority_scores.items(), 
            key=lambda item: item[1], 
            reverse=True
        )

        # "Spend" the energy budget on the highest priority actions
        for head_name, score in sorted_candidates:
            cost = self.head_energy_costs.get(head_name, 0.1)
            if processing_budget >= cost:
                active_heads.append(head_name)
                processing_budget -= cost
            else:
                # Stop if we can't afford the next highest priority action
                break
        
        return active_heads 