"""
Evaluation script for trained Ball Sort Puzzle agents.

Evaluates agent on fixed set of puzzles and computes metrics.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from envs import BallSortEnv
from envs.wrappers import OneHotObservationWrapper
from agents.agent_factory import mask_fn
from utils.config import ConfigManager
import logging

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False

from stable_baselines3 import PPO, A2C, DQN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Ball Sort Puzzle agent"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.zip file)'
    )
    
    parser.add_argument(
        '--algo',
        type=str,
        required=True,
        choices=['ppo', 'a2c', 'dqn'],
        help='Algorithm used for training'
    )
    
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    
    parser.add_argument(
        '--level',
        type=int,
        default=1,
        help='Level to evaluate on'
    )
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='Use deterministic actions'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for evaluation'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: result/evaluations/)'
    )
    
    return parser.parse_args()


def load_model(model_path: str, algo: str):
    """Load trained model."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    algo = algo.lower()
    
    # Try to load as MaskablePPO first if algo is PPO
    if algo == 'ppo':
        if MASKABLE_PPO_AVAILABLE:
            try:
                model = MaskablePPO.load(str(model_path))
                logger.info("Loaded MaskablePPO model")
                return model, True
            except:
                pass
        
        # Fallback to regular PPO
        model = PPO.load(str(model_path))
        logger.info("Loaded PPO model")
        return model, False
    
    elif algo == 'a2c':
        model = A2C.load(str(model_path))
        logger.info("Loaded A2C model")
        return model, False
    
    elif algo == 'dqn':
        model = DQN.load(str(model_path))
        logger.info("Loaded DQN model")
        return model, False
    
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def evaluate_agent(
    model,
    env,
    n_episodes: int,
    deterministic: bool = True,
    render: bool = False,
    use_masking: bool = False,
) -> pd.DataFrame:
    """
    Evaluate agent and collect metrics.
    
    Returns:
        DataFrame with episode-level metrics
    """
    results = []
    
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0
        
        # Reward components
        total_time_penalty = 0.0
        total_purity_reward = 0.0
        total_complete_reward = 0.0
        win_reward = 0.0
        blocked_penalty = 0.0
        
        while not (done or truncated):
            if render:
                env.render()
            
            # Get action
            if use_masking and hasattr(env, 'action_masks'):
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Track reward components
            if 'reward_time' in info:
                total_time_penalty += info['reward_time']
            if 'reward_purity' in info:
                total_purity_reward += info['reward_purity']
            if 'reward_complete' in info:
                total_complete_reward += info['reward_complete']
            if 'reward_win' in info:
                win_reward = info['reward_win']
            if 'penalty_blocked' in info:
                blocked_penalty = info['penalty_blocked']
        
        # Record results
        is_success = info.get('is_success', False)
        level = info.get('level', 1)
        
        results.append({
            'episode': episode,
            'level': level,
            'success': is_success,
            'steps': episode_steps,
            'reward': episode_reward,
            'time_penalty': total_time_penalty,
            'purity_reward': total_purity_reward,
            'complete_reward': total_complete_reward,
            'win_reward': win_reward,
            'blocked_penalty': blocked_penalty,
        })
    
    return pd.DataFrame(results)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = ConfigManager()
    
    # Load model
    model, use_masking = load_model(args.model, args.algo)
    
    # Create environment
    env_kwargs = {
        'n_max': config.get('env.n_max', 14),
        'height': config.get('env.height', 4),
        'initial_level': args.level,
        'max_level': args.level,  # Fixed level for evaluation
        'max_steps': config.get('env.max_steps', 500),
        'seed': args.seed,
        'render_mode': 'human' if args.render else None,
    }
    
    use_one_hot = config.get('env.use_one_hot', True)
    
    base_env = BallSortEnv(**env_kwargs)
    env = OneHotObservationWrapper(base_env) if use_one_hot else base_env
    
    if use_masking:
        env = ActionMasker(env, mask_fn)
    
    logger.info(f"Evaluating on level {args.level} for {args.n_episodes} episodes")
    
    # Evaluate
    results_df = evaluate_agent(
        model=model,
        env=env,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render=args.render,
        use_masking=use_masking,
    )
    
    # Compute summary statistics
    success_rate = results_df['success'].mean()
    mean_steps = results_df['steps'].mean()
    std_steps = results_df['steps'].std()
    mean_reward = results_df['reward'].mean()
    std_reward = results_df['reward'].std()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Level: {args.level}")
    print(f"Episodes: {args.n_episodes}")
    print(f"\nSuccess Rate: {success_rate:.2%}")
    print(f"Mean Steps: {mean_steps:.2f} ± {std_steps:.2f}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("\nReward Breakdown:")
    print(f"  Time Penalty: {results_df['time_penalty'].mean():.2f}")
    print(f"  Purity Reward: {results_df['purity_reward'].mean():.2f}")
    print(f"  Complete Reward: {results_df['complete_reward'].mean():.2f}")
    print(f"  Win Reward: {results_df['win_reward'].mean():.2f}")
    print(f"  Blocked Penalty: {results_df['blocked_penalty'].mean():.2f}")
    print("="*60 + "\n")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.get('paths.evaluations', 'result/evaluations'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_{args.algo}_level{args.level}_{timestamp}.csv"
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")
    
    # Save summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Level: {args.level}\n")
        f.write(f"Episodes: {args.n_episodes}\n")
        f.write(f"Success Rate: {success_rate:.2%}\n")
        f.write(f"Mean Steps: {mean_steps:.2f} ± {std_steps:.2f}\n")
        f.write(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
    
    logger.info(f"Summary saved to: {summary_path}")
    
    env.close()


if __name__ == "__main__":
    main()
