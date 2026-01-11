"""
XAI (Explainable AI) analysis for trained Ball Sort Puzzle agents.

Generates:
- Action probability distributions
- Q-value visualizations (for DQN)
- Action mask overlays
- Integrated Gradients heatmaps (with Captum)
- Trajectory visualizations
- Reward component breakdowns
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import pandas as pd

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
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_env_attr(env, name, default=None):
    """Safely get attribute from env, unwrapping .env if needed."""
    if hasattr(env, name):
        return getattr(env, name)
    if hasattr(env, "env") and hasattr(env.env, name):
        return getattr(env.env, name)
    return default


def decode_obs(obs: np.ndarray) -> np.ndarray:
    """
    Decode observation to int matrix shape (n_max, height).
    Handles one-hot input of shape (n_max, height, channels).
    """
    if obs.ndim == 3:
        decoded = obs.argmax(axis=-1)
        decoded[decoded == 13] = -1  # padding channel
        return decoded
    return obs


def plot_action_logits_and_mask(
    logits: np.ndarray,
    action_mask: np.ndarray,
    n_max: int,
    save_path: Path,
    episode_num: int,
    step_num: int,
):
    """Plot logits heatmap with invalid actions greyed out."""
    logits_2d = logits.reshape(n_max, n_max)
    mask_2d = action_mask.reshape(n_max, n_max)
    masked_logits = logits_2d.copy()
    masked_logits[mask_2d == 0] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    im1 = ax1.imshow(logits_2d, cmap="coolwarm", aspect="auto")
    ax1.set_title(f"Logits (bruts)\nEp {episode_num}, Step {step_num}")
    ax1.set_xlabel("Destination")
    ax1.set_ylabel("Source")
    plt.colorbar(im1, ax=ax1, label="logit")

    im2 = ax2.imshow(masked_logits, cmap="coolwarm", aspect="auto")
    ax2.set_title(f"Logits avec masque\nEp {episode_num}, Step {step_num}")
    ax2.set_xlabel("Destination")
    ax2.set_ylabel("Source")
    plt.colorbar(im2, ax=ax2, label="logit (invalid=NaN)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XAI analysis for Ball Sort Puzzle agents"
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
        '--level',
        type=int,
        default=1,
        help='Level to analyze'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of sample episodes'
    )
    
    parser.add_argument(
        '--use-ig',
        action='store_true',
        help='Use Integrated Gradients (requires captum)'
    )
    
    parser.add_argument(
        '--n-steps-ig',
        type=int,
        default=50,
        help='Number of steps for Integrated Gradients'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def load_model(model_path: str, algo: str):
    """Load trained model."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    algo = algo.lower()
    
    if algo == 'ppo':
        if MASKABLE_PPO_AVAILABLE:
            try:
                model = MaskablePPO.load(str(model_path))
                logger.info("Loaded MaskablePPO model")
                return model, True
            except:
                pass
        
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


def plot_action_distribution(
    action_probs: np.ndarray,
    action_mask: np.ndarray,
    n_max: int,
    save_path: Path,
    episode_num: int,
    step_num: int,
):
    """Plot action probability distribution as heatmap."""
    # Reshape to 2D (source x dest)
    probs_2d = action_probs.reshape(n_max, n_max)
    mask_2d = action_mask.reshape(n_max, n_max)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot probabilities
    im1 = ax1.imshow(probs_2d, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Destination Tube')
    ax1.set_ylabel('Source Tube')
    ax1.set_title(f'Action Probabilities\nEpisode {episode_num}, Step {step_num}')
    plt.colorbar(im1, ax=ax1, label='Probability')
    
    # Plot mask
    im2 = ax2.imshow(mask_2d, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xlabel('Destination Tube')
    ax2.set_ylabel('Source Tube')
    ax2.set_title(f'Valid Actions Mask\nEpisode {episode_num}, Step {step_num}')
    plt.colorbar(im2, ax=ax2, label='Valid (1) / Invalid (0)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_q_values(
    q_values: np.ndarray,
    action_mask: np.ndarray,
    n_max: int,
    save_path: Path,
    episode_num: int,
    step_num: int,
):
    """Plot Q-values as heatmap (for DQN)."""
    q_values_2d = q_values.reshape(n_max, n_max)
    mask_2d = action_mask.reshape(n_max, n_max)
    
    # Mask invalid actions
    masked_q = q_values_2d.copy()
    masked_q[mask_2d == 0] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(masked_q, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Destination Tube')
    ax.set_ylabel('Source Tube')
    ax.set_title(f'Q-Values (Invalid actions masked)\nEpisode {episode_num}, Step {step_num}')
    plt.colorbar(im, ax=ax, label='Q-Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reward_breakdown(
    rewards_df: pd.DataFrame,
    save_path: Path,
):
    """Plot stacked bar chart of reward components."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episodes = rewards_df['episode'].values
    
    # Positive components
    purity = np.maximum(rewards_df['purity_reward'].values, 0)
    complete = rewards_df['complete_reward'].values
    win = rewards_df['win_reward'].values
    
    # Negative components
    time_penalty = np.abs(np.minimum(rewards_df['time_penalty'].values, 0))
    purity_negative = np.abs(np.minimum(rewards_df['purity_reward'].values, 0))
    blocked = np.abs(rewards_df['blocked_penalty'].values)
    
    width = 0.35
    x = np.arange(len(episodes))
    
    # Positive stack
    ax.bar(x - width/2, purity, width, label='Purity +', color='green', alpha=0.7)
    ax.bar(x - width/2, complete, width, bottom=purity, label='Complete +', color='blue', alpha=0.7)
    ax.bar(x - width/2, win, width, bottom=purity+complete, label='Win +', color='gold', alpha=0.7)
    
    # Negative stack
    ax.bar(x + width/2, -time_penalty, width, label='Time -', color='orange', alpha=0.7)
    ax.bar(x + width/2, -purity_negative, width, bottom=-time_penalty, label='Purity -', color='red', alpha=0.7)
    ax.bar(x + width/2, -blocked, width, bottom=-time_penalty-purity_negative, label='Blocked -', color='darkred', alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Components')
    ax.set_title('Reward Breakdown by Episode')
    ax.set_xticks(x)
    ax.set_xticklabels(episodes)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_trajectory(
    trajectory: list,
    save_path: Path,
    episode_num: int,
):
    """Plot state trajectory visualization."""
    fig, axes = plt.subplots(1, len(trajectory), figsize=(4*len(trajectory), 6))
    
    if len(trajectory) == 1:
        axes = [axes]
    
    for idx, (state, action, reward) in enumerate(trajectory):
        ax = axes[idx]
        
        # Visualize state
        n_tubes = (state[:, 0] != -1).sum()
        state_display = state[:n_tubes, :]
        
        # Create color-coded display
        im = ax.imshow(state_display.T, cmap='tab20', aspect='auto', vmin=-1, vmax=12)
        ax.set_xlabel('Tube')
        ax.set_ylabel('Height')
        ax.set_title(f'Step {idx}\nAction: {action}\nReward: {reward:.2f}')
        ax.set_xticks(range(n_tubes))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def integrated_gradients_analysis(
    model,
    obs: np.ndarray,
    action: int,
    n_steps: int = 50,
):
    """
    Compute Integrated Gradients attribution.
    
    Note: For discrete observations, we convert to one-hot encoding.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        logger.warning("Captum not installed, skipping Integrated Gradients")
        return None
    
    # Flatten observation (already float from one-hot wrapper)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    obs_tensor.requires_grad_(True)

    def forward_func(x):
        if hasattr(model, "policy") and hasattr(model.policy, "get_distribution"):
            dist = model.policy.get_distribution(x)
            logits = dist.distribution.logits
            return logits[:, action]
        elif hasattr(model, "q_net"):
            q_values = model.q_net(x)
            return q_values[:, action]
        else:
            return torch.sum(x)  # fallback

    try:
        ig = IntegratedGradients(forward_func)
        baseline = torch.zeros_like(obs_tensor)
        attributions = ig.attribute(obs_tensor, baseline, n_steps=n_steps)
        return attributions.detach().cpu().numpy()
    except Exception as e:
        logger.error(f"Integrated Gradients failed: {e}")
        return None


def analyze_episode(
    model,
    env,
    episode_num: int,
    output_dir: Path,
    use_masking: bool,
    use_ig: bool,
    n_steps_ig: int,
    algo: str,
):
    """Analyze a single episode."""
    obs, info = env.reset()
    done = False
    truncated = False
    
    trajectory = []
    rewards_data = {
        'episode': episode_num,
        'time_penalty': 0.0,
        'purity_reward': 0.0,
        'complete_reward': 0.0,
        'win_reward': 0.0,
        'blocked_penalty': 0.0,
    }
    entropies = []
    values = []
    
    step_num = 0
    
    while not (done or truncated) and step_num < 10:  # Limit visualization steps
        # Get action mask
        if hasattr(env, 'action_masks'):
            action_mask = env.action_masks()
        else:
            action_mask = np.ones(env.action_space.n)
        n_max = _get_env_attr(env, "n_max", 14)
        
        # Get action and probabilities
        if use_masking and hasattr(model, 'predict'):
            if algo == 'dqn':
                # DQN: get Q-values
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    q_values = model.q_net(obs_tensor).cpu().numpy()[0]
                    # No entropy/value for DQN here
                action, _ = model.predict(obs, deterministic=True)
                
                # Plot Q-values
                q_plot_path = output_dir / f"q_values_ep{episode_num}_step{step_num}.png"
                plot_q_values(q_values, action_mask, n_max, q_plot_path, episode_num, step_num)
                
            else:
                # PPO/A2C: get action probabilities
                action, _ = model.predict(obs, action_masks=action_mask if use_masking else None, deterministic=False)
                
                # Get action probabilities
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    if hasattr(model.policy, 'get_distribution'):
                        dist = model.policy.get_distribution(obs_tensor)
                        action_probs = dist.distribution.probs.cpu().numpy()[0]
                        # Entropy and value (critic)
                        if hasattr(dist.distribution, "entropy"):
                            ent = dist.distribution.entropy().cpu().numpy()[0]
                            entropies.append(ent)
                        if hasattr(model.policy, "predict_values"):
                            val = model.policy.predict_values(obs_tensor).cpu().numpy()[0][0]
                            values.append(val)
                    else:
                        action_probs = np.ones(env.action_space.n) / env.action_space.n
                
                # Plot action distribution
                action_plot_path = output_dir / f"action_dist_ep{episode_num}_step{step_num}.png"
                plot_action_distribution(action_probs, action_mask, n_max, action_plot_path, episode_num, step_num)
                # Plot logits vs mask
                if hasattr(model.policy, "get_distribution"):
                    logits = dist.distribution.logits.cpu().numpy()[0]
                    logits_mask_path = output_dir / f"logits_mask_ep{episode_num}_step{step_num}.png"
                    plot_action_logits_and_mask(logits, action_mask, n_max, logits_mask_path, episode_num, step_num)
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        # Integrated Gradients (expensive, do sparingly)
        if use_ig and step_num == 0:
            ig_path = output_dir / f"ig_ep{episode_num}_step{step_num}.png"
            attributions = integrated_gradients_analysis(model, obs, action, n_steps_ig)
            if attributions is not None:
                try:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(attributions.reshape(-1, 1), cmap='RdBu', aspect='auto')
                    plt.title(f'Integrated Gradients Attribution\nEpisode {episode_num}, Step {step_num}')
                    plt.colorbar(label='Attribution')
                    plt.savefig(ig_path, dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.error(f"IG plotting failed: {e}")
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Record trajectory (decoded for plotting)
        trajectory.append((decode_obs(obs).copy(), action, reward))
        
        # Accumulate rewards
        rewards_data['time_penalty'] += info.get('reward_time', 0.0)
        rewards_data['purity_reward'] += info.get('reward_purity', 0.0)
        rewards_data['complete_reward'] += info.get('reward_complete', 0.0)
        rewards_data['win_reward'] = info.get('reward_win', 0.0)
        rewards_data['blocked_penalty'] = info.get('penalty_blocked', 0.0)
        
        obs = next_obs
        step_num += 1
    
    # Plot trajectory
    if trajectory:
        traj_path = output_dir / f"trajectory_ep{episode_num}.png"
        plot_trajectory(trajectory[:5], traj_path, episode_num)  # First 5 steps
    
    return rewards_data, entropies, values


def main():
    """Main XAI analysis function."""
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
        'max_level': args.level,
        'max_steps': config.get('env.max_steps', 500),
        'seed': args.seed,
    }
    
    base_env = BallSortEnv(**env_kwargs)
    env = OneHotObservationWrapper(base_env) if config.get('env.use_one_hot', True) else base_env
    
    if use_masking and ActionMasker is not None:
        env = ActionMasker(env, mask_fn)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get('paths.xai', 'result/xai')) / f"{args.algo}_level{args.level}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting XAI analysis for {args.n_samples} episodes")
    logger.info(f"Output directory: {output_dir}")
    
    # Analyze episodes
    all_rewards = []
    all_entropies = []
    all_values = []
    
    for ep in range(args.n_samples):
        logger.info(f"Analyzing episode {ep + 1}/{args.n_samples}")
        
        rewards_data, ent_list, val_list = analyze_episode(
            model=model,
            env=env,
            episode_num=ep,
            output_dir=output_dir,
            use_masking=use_masking,
            use_ig=args.use_ig and (ep < 3),  # Only first 3 episodes for IG
            n_steps_ig=args.n_steps_ig,
            algo=args.algo,
        )
        
        all_rewards.append(rewards_data)
        if ent_list:
            all_entropies.append(ent_list)
        if val_list:
            all_values.append(val_list)
    
    # Create reward breakdown plot
    rewards_df = pd.DataFrame(all_rewards)
    reward_breakdown_path = output_dir / "reward_breakdown.png"
    plot_reward_breakdown(rewards_df, reward_breakdown_path)
    
    # Save rewards data
    rewards_csv_path = output_dir / "rewards_breakdown.csv"
    rewards_df.to_csv(rewards_csv_path, index=False)
    
    logger.info(f"\nXAI Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Action distributions/Q-values: action_dist_*.png or q_values_*.png")
    logger.info(f"  - Trajectories: trajectory_*.png")
    logger.info(f"  - Reward breakdown: reward_breakdown.png")
    if args.use_ig:
        logger.info(f"  - Integrated Gradients: ig_*.png")
    if all_entropies:
        logger.info("  - Entropy/value arrays recorded (not plotted)")
    
    env.close()


if __name__ == "__main__":
    main()
