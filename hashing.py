
import numpy as np
from collections import OrderedDict

class LSHSystem:
    """
    Implements a Locality-Sensitive Hashing (LSH) system to efficiently
    find similar high-dimensional vectors (e.g., model states).
    
    This implementation uses random projection to create hash signatures.
    """
    def __init__(self, input_dim, num_hashes):
        """
        Initializes the LSH system.
        """
        self.input_dim = input_dim
        self.num_hashes = num_hashes
        self.random_planes = np.random.randn(self.num_hashes, self.input_dim)
        
        # Long-term persistent memory
        self.master_hashes = []
        self.master_hash_counts = {}
        self.master_hash_data = {}
        
        # Short-term volatile memory for recently demoted hashes
        self.demoted_cache_size = 50
        self.demoted_cache = OrderedDict()
        
        # Cache for novel hashes not yet promoted
        self.novel_hash_cache = {}

    def hash(self, input_vector):
        """
        Computes the LSH signature for a given input vector.

        Args:
            input_vector (np.array): A vector of shape (input_dim,).

        Returns:
            str: A binary string representing the hash signature.
        """
        # Determine which side of each plane the vector lies on.
        # A dot product > 0 means it's on the "positive" side.
        projections = np.dot(self.random_planes, input_vector)
        # Convert booleans (True/False) to a binary string '10110...'
        return "".join(['1' if p > 0 else '0' for p in projections])

    def _hamming_distance(self, hash1, hash2):
        """Calculates the Hamming distance (number of differing bits) between two hash strings."""
        if len(hash1) != len(hash2):
            raise ValueError("Hashes must have the same length.")
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def set_master_hashes(self, master_hashes_data):
        """
        Sets the master hashes and their associated data (like optimal_depth).

        Args:
            master_hashes_data (dict): A dictionary where keys are hash strings
                                     and values are dicts containing their data.
                                     e.g., {"hash1": {"count": 10, "optimal_depth": 32}}
        """
        self.master_hashes = list(master_hashes_data.keys())
        self.master_hash_counts = {h: d.get("count", 0) for h, d in master_hashes_data.items()}
        # This new dictionary will store all data, including optimal_depth
        self.master_hash_data = master_hashes_data
        print(f"Master hashes set. Total: {len(self.master_hashes)}")

    def _promote_hash(self, novel_hash):
        """Internal helper to promote a novel hash to a master hash."""
        if novel_hash not in self.novel_hash_cache:
            return

        novel_data = self.novel_hash_cache[novel_hash]
        novel_count = novel_data.get("count", 0)

        self.master_hashes.append(novel_hash)
        self.master_hash_counts[novel_hash] = novel_count
        
        # Transfer all relevant data from the novel cache to the master data
        self.master_hash_data[novel_hash] = {
            "count": novel_count,
            "first_seen_tick": novel_data.get("first_seen_tick"),
            "last_seen_tick": novel_data.get("last_seen_tick"),
            "optimal_depths": {} # Initialize empty depths
        }
        
        # Remove from the novel cache
        del self.novel_hash_cache[novel_hash]

    def _demote_hash(self, master_hash_to_demote):
        """Internal helper to demote a master hash."""
        if master_hash_to_demote not in self.master_hash_data:
            return

        demoted_master_data = self.master_hash_data[master_hash_to_demote]
        self.demoted_cache[master_hash_to_demote] = demoted_master_data
        
        # Remove from master lists
        self.master_hashes.remove(master_hash_to_demote)
        del self.master_hash_counts[master_hash_to_demote]
        del self.master_hash_data[master_hash_to_demote]

        # Ensure the demoted cache does not exceed its size limit
        if len(self.demoted_cache) > self.demoted_cache_size:
            self.demoted_cache.popitem(last=False) # Remove the oldest item

    def consider_promotion(self, novel_hash):
        """
        Checks if a novel hash should be promoted, handling both filling
        and replacement scenarios.
        """
        novel_info = self.novel_hash_cache.get(novel_hash)
        # Add a small threshold to prevent one-off hashes from being promoted
        if not novel_info or novel_info.get("count", 0) < 3:
            return

        # --- Phase 1: Filling the master hash list ---
        if len(self.master_hashes) < 100:
            print(f"PROMOTION (Filling): Promoting novel hash {novel_hash[:8]}...")
            self._promote_hash(novel_hash)
            return

        # --- Phase 2: Replacing the least frequent master (once full) ---
        # Find the least frequent hash in the master set
        # Using .get() provides a default if the key somehow doesn't exist
        least_frequent_master = min(self.master_hash_counts, key=self.master_hash_counts.get)
        least_frequent_count = self.master_hash_counts.get(least_frequent_master, 0)

        # Check if the novel hash is more frequent
        if novel_info.get("count", 0) > least_frequent_count:
            print(f"PROMOTION (Replacing): Novel {novel_hash[:8]} ({novel_info['count']}) replaces Master {least_frequent_master[:8]} ({least_frequent_count})")
            self._demote_hash(least_frequent_master)
            self._promote_hash(novel_hash)

    def query_and_update(self, input_vector, current_tick):
        """
        Hashes the input vector, compares it to master and demoted hashes,
        and updates counts.

        Args:
            input_vector (np.array): The vector to query.
            current_tick (int): The current engine tick for logging.

        Returns:
            dict: A result dictionary indicating the hash status.
        """
        current_hash = self.hash(input_vector)
        max_distance = int(self.num_hashes * 0.05)

        # 1. Check against master hashes
        for i, master_hash in enumerate(self.master_hashes):
            if self._hamming_distance(current_hash, master_hash) <= max_distance:
                self.master_hash_counts[master_hash] += 1
                # Update the last_seen_tick for this master hash
                if master_hash in self.master_hash_data:
                    self.master_hash_data[master_hash]["last_seen_tick"] = current_tick
                    self.master_hash_data[master_hash]["count"] = self.master_hash_counts[master_hash]
                # Retrieve all data for this master, including optimal_depths
                master_data = self.master_hash_data.get(master_hash, {})
                return {
                    "status": "master",
                    "master_id": i,
                    "hash": master_hash,
                    "optimal_depths": master_data.get("optimal_depths", {}) # <-- CHANGED
                }

        # 2. Check against the short-term demoted cache
        for demoted_hash in self.demoted_cache:
            if self._hamming_distance(current_hash, demoted_hash) <= max_distance:
                return {"status": "demoted", "hash": demoted_hash}
        
        # 3. If no match, process as a novel hash
        if current_hash not in self.novel_hash_cache:
            self.novel_hash_cache[current_hash] = {"count": 0, "first_seen_tick": current_tick}
        
        self.novel_hash_cache[current_hash]["count"] += 1
        # Update last_seen_tick for novel hashes
        self.novel_hash_cache[current_hash]["last_seen_tick"] = current_tick
        
        # Store the count before potential promotion
        current_count = self.novel_hash_cache[current_hash]["count"]
        
        # 4. Consider promotion after updating the count
        self.consider_promotion(current_hash)
        
        # Check if the hash was promoted (removed from novel cache)
        if current_hash in self.novel_hash_cache:
            return {"status": "novel", "hash": current_hash, "count": current_count}
        else:
            # Hash was promoted to master, return master status
            return {"status": "master", "hash": current_hash, "count": current_count}

 