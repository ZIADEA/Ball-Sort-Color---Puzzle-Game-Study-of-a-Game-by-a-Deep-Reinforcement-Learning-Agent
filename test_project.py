"""
Quick test script to validate project setup.

Tests:
1. Environment creation and basic operations
2. Action masking
3. Episode completion
4. Config loading
5. Directory structure
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from envs import BallSortEnv
from utils.config import ConfigManager, setup_directories
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment():
    """Test environment creation and basic operations."""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation")
    print("="*60)
    
    try:
        env = BallSortEnv(
            n_max=14,
            height=4,
            initial_level=1,
            max_level=5,
            seed=42
        )
        
        obs, info = env.reset()
        
        assert obs.shape == (14, 4), f"Wrong obs shape: {obs.shape}"
        assert obs.dtype == np.int32, f"Wrong dtype: {obs.dtype}"
        assert 'level' in info, "Missing level in info"
        
        print("✓ Environment created successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        print(f"  Level: {info['level']}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_action_masking():
    """Test action masking functionality."""
    print("\n" + "="*60)
    print("TEST 2: Action Masking")
    print("="*60)
    
    try:
        env = BallSortEnv(initial_level=1, seed=42)
        env.reset()
        
        # Get action masks
        mask = env.action_masks()
        
        assert mask.shape == (env.action_space.n,), f"Wrong mask shape: {mask.shape}"
        assert mask.dtype == np.int8, f"Wrong mask dtype: {mask.dtype}"
        assert np.all((mask == 0) | (mask == 1)), "Mask values should be 0 or 1"
        
        valid_actions = np.sum(mask)
        total_actions = env.action_space.n
        
        print("✓ Action masking works")
        print(f"  Valid actions: {valid_actions}/{total_actions}")
        print(f"  Mask shape: {mask.shape}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Action masking test failed: {e}")
        return False


def test_episode():
    """Test running a complete episode."""
    print("\n" + "="*60)
    print("TEST 3: Episode Execution")
    print("="*60)
    
    try:
        env = BallSortEnv(initial_level=1, max_steps=100, seed=42)
        obs, info = env.reset()
        
        done = False
        truncated = False
        steps = 0
        total_reward = 0
        
        while not (done or truncated) and steps < 10:
            # Get valid actions
            mask = env.action_masks()
            valid_actions = np.where(mask == 1)[0]
            
            if len(valid_actions) == 0:
                break
            
            # Take random valid action
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
        
        print("✓ Episode runs successfully")
        print(f"  Steps taken: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Episode ended: done={done}, truncated={truncated}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Episode test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("TEST 4: Configuration System")
    print("="*60)
    
    try:
        config = ConfigManager()
        
        # Test get method
        n_max = config.get('env.n_max')
        assert n_max == 14, f"Wrong n_max: {n_max}"
        
        # Test nested get
        learning_rate = config.get('ppo.learning_rate')
        assert learning_rate is not None, "Missing PPO learning rate"
        
        print("✓ Configuration loads successfully")
        print(f"  env.n_max: {n_max}")
        print(f"  ppo.learning_rate: {learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_directories():
    """Test directory structure."""
    print("\n" + "="*60)
    print("TEST 5: Directory Structure")
    print("="*60)
    
    try:
        required_dirs = [
            "result/models",
            "result/logs",
            "result/evaluations",
            "result/xai",
            "result/episodes",
            "result/configs",
        ]
        
        all_exist = True
        for dir_path in required_dirs:
            path = Path(dir_path)
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {dir_path}")
            all_exist = all_exist and exists
        
        if all_exist:
            print("✓ All directories exist")
            return True
        else:
            print("✗ Some directories missing")
            return False
        
    except Exception as e:
        print(f"✗ Directory test failed: {e}")
        return False


def test_imports():
    """Test all critical imports."""
    print("\n" + "="*60)
    print("TEST 6: Dependencies")
    print("="*60)
    
    imports = {
        'stable_baselines3': None,
        'gymnasium': None,
        'torch': None,
        'numpy': None,
        'pandas': None,
        'matplotlib': None,
        'yaml': None,
    }
    
    optional_imports = {
        'sb3_contrib': 'MaskablePPO',
        'pygame': 'PyGame rendering',
        'captum': 'Integrated Gradients',
        'imageio': 'GIF recording',
    }
    
    try:
        # Test required imports
        all_ok = True
        for module in imports:
            try:
                __import__(module)
                print(f"  ✓ {module}")
            except ImportError:
                print(f"  ✗ {module} (REQUIRED)")
                all_ok = False
        
        # Test optional imports
        print("\n  Optional dependencies:")
        for module, feature in optional_imports.items():
            try:
                __import__(module)
                print(f"  ✓ {module} ({feature})")
            except ImportError:
                print(f"  ⚠ {module} ({feature}) - optional")
        
        if all_ok:
            print("\n✓ All required dependencies available")
            return True
        else:
            print("\n✗ Some required dependencies missing")
            return False
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("BALL SORT PUZZLE RL - PROJECT VALIDATION")
    print("="*60)
    
    tests = [
        ("Dependencies", test_imports),
        ("Environment", test_environment),
        ("Action Masking", test_action_masking),
        ("Episode Execution", test_episode),
        ("Configuration", test_config),
        ("Directories", test_directories),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("  1. See commande.md for all available commands")
        print("  2. Start training: python train.py --algo ppo --timesteps 10000")
        print("  3. Monitor with: tensorboard --logdir result/logs")
    else:
        print("\n⚠ Some tests failed. Please check errors above.")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Check that all directories exist")
    
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
