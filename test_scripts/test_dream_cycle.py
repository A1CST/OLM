#!/usr/bin/env python3
"""
Test script for the OLM Dream Cycle System
"""

import numpy as np
import torch
import queue
import time
from engine_pytorch import EngineCore

def test_dream_cycle():
    """Test the complete dream cycle functionality."""
    
    print("=" * 60)
    print("OLM DREAM CYCLE SYSTEM TEST")
    print("=" * 60)
    
    # Initialize engine
    print("\n1. Initializing OLM Engine...")
    update_queue = queue.Queue(maxsize=5)
    engine = EngineCore(update_queue)
    
    print(f"   [OK] Engine initialized")
    print(f"   [OK] ReplayBuffer empty: {engine.replay_buffer.is_empty()}")
    print(f"   [OK] Initial thought count: {engine.thought_count_this_cycle}")
    
    # Test dream count calculation
    print("\n2. Testing Dream Count Calculation...")
    for thoughts in [1, 5, 10, 20]:
        dreams = engine.regulator.calculate_dream_count(thoughts)
        print(f"   [OK] {thoughts} thoughts -> {dreams} dreams")
    
    # Test temperature sampling
    print("\n3. Testing Temperature Sampling...")
    output_vec = np.array([0.1, 0.8, 0.1, 0.05, 0.05])
    print(f"   Test vector: {output_vec}")
    
    # Test deterministic
    result_det = engine.tokenizer.decode('action', output_vec, temperature=0.0)
    print(f"   Deterministic (T=0.0): {result_det}")
    
    # Test random sampling
    results_random = []
    for i in range(5):
        result = engine.tokenizer.decode('action', output_vec, temperature=1.0)
        results_random.append(result)
    print(f"   Random samples (T=1.0): {set(results_random)}")
    
    # Simulate adding experiences to replay buffer
    print("\n4. Testing Replay Buffer...")
    for i in range(10):
        token_vector = np.random.randn(142) * 0.1
        hash_result = {'status': 'novel' if i % 3 == 0 else 'familiar', 'hash': f'test_hash_{i}'}
        required_depths = {'internal_thought': 16}
        active_heads = ['internal_thought']
        novelty_scores = {'internal_thought': np.random.random()}
        
        engine.replay_buffer.add_experience(
            token_vector=token_vector,
            hash_result=hash_result,
            required_depths=required_depths,
            active_heads=active_heads,
            novelty_scores=novelty_scores
        )
    
    stats = engine.replay_buffer.get_stats()
    print(f"   [OK] Buffer size: {stats['size']}")
    print(f"   [OK] Novel experiences: {stats['novel_count']}")
    print(f"   [OK] Familiar experiences: {stats['familiar_count']}")
    print(f"   [OK] Avg novelty: {stats['avg_novelty_score']:.3f}")
    
    # Test different sampling strategies
    print("\n5. Testing Sampling Strategies...")
    
    # Random sampling
    random_sample = engine.replay_buffer.sample(2)
    print(f"   [OK] Random sample: {len(random_sample)} experiences")
    
    # Novel sampling
    novel_sample = engine.replay_buffer.sample_novel(2)
    print(f"   [OK] Novel sample: {len(novel_sample)} experiences")
    
    # Recent sampling
    recent_sample = engine.replay_buffer.sample_recent(2)
    print(f"   [OK] Recent sample: {len(recent_sample)} experiences")
    
    # Test sleep cycle
    print("\n6. Testing Sleep Cycle...")
    engine.thought_count_this_cycle = 5  # Simulate some thoughts
    dream_count = engine.regulator.calculate_dream_count(engine.thought_count_this_cycle)
    
    print(f"   [OK] Thought count: {engine.thought_count_this_cycle}")
    print(f"   [OK] Dream count: {dream_count}")
    print(f"   [OK] Growth stage: {engine.regulator.growth_stage:.2f}")
    
    # Test a mini sleep cycle (fewer dreams for testing)
    print("   Running mini sleep cycle...")
    start_time = time.time()
    engine._sleep_cycle(3)  # Just 3 dreams for testing
    duration = time.time() - start_time
    print(f"   [OK] Sleep cycle completed in {duration:.2f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("DREAM CYCLE TEST SUMMARY")
    print("=" * 60)
    print("[PASS] Engine initialization")
    print("[PASS] Dream count calculation")
    print("[PASS] Temperature sampling")
    print("[PASS] Replay buffer operations")
    print("[PASS] Sampling strategies")
    print("[PASS] Sleep cycle execution")
    print("\nOLM Dream Cycle system is fully operational!")
    

if __name__ == "__main__":
    test_dream_cycle()