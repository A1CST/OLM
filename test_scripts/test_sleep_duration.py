#!/usr/bin/env python3
"""
Test script for Growth Stage-Based Sleep Duration System
"""

from regulator import Regulator

def test_sleep_durations():
    """Test sleep duration calculations across different growth stages."""
    
    print("=" * 60)
    print("GROWTH STAGE-BASED SLEEP DURATION TEST")
    print("=" * 60)
    
    # Test various growth stages
    test_stages = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    
    print(f"{'Growth Stage':<12} {'Sleep Ticks':<12} {'Sleep Seconds':<15} {'Expected Range'}")
    print("-" * 60)
    
    for stage in test_stages:
        regulator = Regulator()
        regulator.growth_stage = stage
        regulator._update_parameters()
        
        sleep_ticks = regulator.sleep_duration_ticks
        sleep_seconds = sleep_ticks / 20.0  # Assuming 20 TPS
        
        # Determine expected range
        if stage <= 1.0:
            expected = "25-30s"
        elif stage <= 5.0:
            expected = "20-25s"
        else:
            expected = "15-20s"
        
        print(f"{stage:<12.1f} {sleep_ticks:<12} {sleep_seconds:<15.1f} {expected}")
    
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    # Validate requirements
    max_sleep = max([Regulator() for _ in range(10)], 
                   key=lambda r: (setattr(r, 'growth_stage', 0.1), r._update_parameters(), r.sleep_duration_ticks)[-1]).sleep_duration_ticks
    min_sleep = max([Regulator() for _ in range(10)], 
                   key=lambda r: (setattr(r, 'growth_stage', 20.0), r._update_parameters(), -r.sleep_duration_ticks)[-1])
    min_sleep._update_parameters()
    min_sleep_ticks = min_sleep.sleep_duration_ticks
    
    print(f"Max sleep duration: {max_sleep} ticks ({max_sleep/20:.1f}s)")
    print(f"Min sleep duration: {min_sleep_ticks} ticks ({min_sleep_ticks/20:.1f}s)")
    print(f"Requirement: Max 30s (600 ticks): {'[OK]' if max_sleep <= 600 else '[FAIL]'}")
    print(f"Requirement: Min 15s (300 ticks): {'[OK]' if min_sleep_ticks >= 300 else '[FAIL]'}")
    print(f"Lower growth = longer sleep: {'[OK]' if max_sleep > min_sleep_ticks else '[FAIL]'}")


if __name__ == "__main__":
    test_sleep_durations()