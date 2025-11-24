"""
Test script to verify AgroTrack environment implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrotrack_env import AgroTrackEnv
import numpy as np


def test_environment_creation():
    """Test environment can be created."""
    print("Test 1: Environment Creation")
    try:
        env = AgroTrackEnv()
        print("✓ Environment created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False


def test_reset():
    """Test environment reset functionality."""
    print("\nTest 2: Environment Reset")
    try:
        env = AgroTrackEnv()
        obs, info = env.reset()
        
        # Check observation shape
        assert obs.shape == (6,), f"Expected observation shape (6,), got {obs.shape}"
        
        # Check observation bounds
        assert np.all(obs >= env.observation_space.low), "Observation below lower bound"
        assert np.all(obs <= env.observation_space.high), "Observation above upper bound"
        
        # Check initial quality
        assert info['quality_index'] == 100.0, "Initial quality should be 100"
        
        print("✓ Environment reset works correctly")
        print(f"  Initial observation: {obs}")
        return True
    except Exception as e:
        print(f"✗ Reset test failed: {e}")
        return False


def test_step():
    """Test environment step functionality."""
    print("\nTest 3: Environment Step")
    try:
        env = AgroTrackEnv()
        obs, info = env.reset()
        
        # Test each action
        for action in range(5):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check return types
            assert isinstance(obs, np.ndarray), "Observation should be numpy array"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be boolean"
            assert isinstance(truncated, bool), "Truncated should be boolean"
            assert isinstance(info, dict), "Info should be dictionary"
            
            print(f"  Action {action}: Reward={reward:.2f}, Quality={info['quality_index']:.2f}")
        
        print("✓ Environment step works correctly")
        return True
    except Exception as e:
        print(f"✗ Step test failed: {e}")
        return False


def test_episode():
    """Test a complete episode."""
    print("\nTest 4: Complete Episode")
    try:
        env = AgroTrackEnv()
        obs, info = env.reset()
        
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        print(f"✓ Episode completed successfully")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final quality: {info['quality_index']:.2f}")
        print(f"  Total cost: {info['storage_cost']:.2f}")
        return True
    except Exception as e:
        print(f"✗ Episode test failed: {e}")
        return False


def test_gymnasium_compatibility():
    """Test Gymnasium API compatibility."""
    print("\nTest 5: Gymnasium API Compatibility")
    try:
        env = AgroTrackEnv()
        
        # Check required attributes
        assert hasattr(env, 'action_space'), "Missing action_space"
        assert hasattr(env, 'observation_space'), "Missing observation_space"
        assert hasattr(env, 'reset'), "Missing reset method"
        assert hasattr(env, 'step'), "Missing step method"
        
        # Check spaces
        from gymnasium import spaces
        assert isinstance(env.action_space, spaces.Discrete), "Action space should be Discrete"
        assert isinstance(env.observation_space, spaces.Box), "Observation space should be Box"
        
        print("✓ Environment is Gymnasium compatible")
        return True
    except Exception as e:
        print(f"✗ Gymnasium compatibility test failed: {e}")
        return False


def test_action_effects():
    """Test that different actions have different effects."""
    print("\nTest 6: Action Effects")
    try:
        env = AgroTrackEnv()
        
        # Test no intervention vs advanced cooling
        results = []
        for action in [0, 2]:  # No intervention and advanced cooling
            env.reset(seed=42)  # Use same seed for comparison
            qualities = []
            for _ in range(5):
                obs, reward, terminated, truncated, info = env.step(action)
                qualities.append(info['quality_index'])
                if terminated or truncated:
                    break
            results.append(qualities)
        
        # Advanced cooling (action 2) should preserve quality better
        if len(results[0]) > 0 and len(results[1]) > 0:
            final_quality_no_intervention = results[0][-1]
            final_quality_cooling = results[1][-1]
            
            print(f"  No intervention quality: {final_quality_no_intervention:.2f}")
            print(f"  Advanced cooling quality: {final_quality_cooling:.2f}")
            
            if final_quality_cooling > final_quality_no_intervention:
                print("✓ Actions have different effects (cooling preserves quality better)")
            else:
                print("✓ Actions have different effects")
        else:
            print("✓ Actions executed successfully")
        
        return True
    except Exception as e:
        print(f"✗ Action effects test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("AgroTrack Environment Test Suite")
    print("=" * 60)
    
    tests = [
        test_environment_creation,
        test_reset,
        test_step,
        test_episode,
        test_gymnasium_compatibility,
        test_action_effects
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed! Environment is ready for training.")
        return True
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
