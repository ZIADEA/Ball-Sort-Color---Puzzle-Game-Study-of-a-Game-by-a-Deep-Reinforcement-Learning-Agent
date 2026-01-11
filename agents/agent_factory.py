"""
Agent wrappers for different RL algorithms.

Provides unified interface for training PPO, A2C, DQN with action masking.
"""

from typing import Optional, Dict, Any, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import logging

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    logging.warning("sb3-contrib not available, MaskablePPO will not work")

try:
    from envs.wrappers import OneHotObservationWrapper
    ONE_HOT_AVAILABLE = True
except ImportError:
    ONE_HOT_AVAILABLE = False
    logging.warning("OneHotObservationWrapper not available; using raw observations")

logger = logging.getLogger(__name__)


class InvalidActionWrapper(gym.Wrapper):
    """
    Wrapper to handle invalid actions for algorithms without native masking.
    
    Penalizes invalid actions and prevents them from being executed.
    """
    
    def __init__(self, env: gym.Env, penalty: float = -1.0):
        super().__init__(env)
        self.penalty = penalty
        
    def step(self, action):
        # Get action mask
        if hasattr(self.env, 'action_masks'):
            mask = self.env.action_masks()
            
            # Check if action is invalid
            if mask[action] == 0:
                # Return penalty without changing state
                obs = self.env.state.copy() if hasattr(self.env, 'state') else None
                return obs, self.penalty, False, False, {"invalid_action": True}
        
        # Action is valid, execute normally
        return self.env.step(action)


def mask_fn(env: gym.Env) -> np.ndarray:
    """Masking function for ActionMasker wrapper."""
    masks = env.action_masks()
    # Ensure boolean mask and never all-zero to keep MaskablePPO stable
    masks = np.asarray(masks, dtype=bool)
    if masks.size == 0 or not masks.any():
        masks = np.ones_like(masks, dtype=bool)
    return masks


def create_env(
    env_kwargs: Dict[str, Any],
    use_masking: bool = True,
    rank: int = 0,
    seed: int = 0,
    use_one_hot: bool = True,
) -> gym.Env:
    """
    Create a single environment instance.
    
    Args:
        env_kwargs: Environment initialization arguments
        use_masking: Whether to use action masking
        rank: Environment rank (for parallel envs)
        seed: Random seed
    
    Returns:
        Wrapped environment
    """
    def _init():
        from envs import BallSortEnv
        
        env = BallSortEnv(**env_kwargs, seed=seed + rank)
        
        if use_one_hot and ONE_HOT_AVAILABLE:
            env = OneHotObservationWrapper(env)
        
        if use_masking:
            env = ActionMasker(env, mask_fn)
        
        env = Monitor(env)
        return env
    
    return _init


def make_vec_env(
    n_envs: int,
    env_kwargs: Dict[str, Any],
    use_masking: bool = True,
    seed: int = 0,
    use_one_hot: bool = True,
    vec_env_cls: Optional[type] = None,
) -> VecMonitor:
    """
    Create vectorized environment.
    
    Args:
        n_envs: Number of parallel environments
        env_kwargs: Environment initialization arguments
        use_masking: Whether to use action masking
        seed: Random seed
        vec_env_cls: VecEnv class (SubprocVecEnv or DummyVecEnv)
    
    Returns:
        Vectorized environment
    """
    if vec_env_cls is None:
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    
    env_fns = [
        create_env(env_kwargs, use_masking, rank, seed, use_one_hot)
        for rank in range(n_envs)
    ]
    
    vec_env = vec_env_cls(env_fns)
    vec_env = VecMonitor(vec_env)
    
    return vec_env


class CurriculumCallback(BaseCallback):
    """
    Callback to manage curriculum learning progression.
    
    Tracks success rate across ALL parallel environments and increases
    difficulty level when agent consistently achieves high success rate.
    """
    
    def __init__(
        self,
        vec_env,
        curriculum_window: int = 500,
        curriculum_threshold: float = 0.95,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.curriculum_window = curriculum_window
        self.curriculum_threshold = curriculum_threshold
        self.current_level = 1
        self.episode_results = []  # Global across all environments
        self.current_episode_success = [False] * vec_env.num_envs  # Track if current episode succeeded
        
    def _on_step(self) -> bool:
        # Track is_success for current episodes (before Monitor wraps it)
        if len(self.locals.get('infos', [])) > 0:
            for idx, info in enumerate(self.locals['infos']):
                # Update success status if present in info
                if 'is_success' in info:
                    self.current_episode_success[idx] = bool(info['is_success'])
                
                # Check if episode just finished
                if 'episode' in info:
                    # Record the success status for this completed episode
                    self.episode_results.append(
                        1.0 if self.current_episode_success[idx] else 0.0
                    )
                    # Reset for next episode
                    self.current_episode_success[idx] = False
                    
                    # Check if we should increase curriculum level
                    if len(self.episode_results) >= self.curriculum_window:
                        recent_success = np.mean(
                            self.episode_results[-self.curriculum_window:]
                        )
                        
                        if recent_success >= self.curriculum_threshold:
                            # Get current max level from environments
                            max_level = getattr(
                                self.vec_env.get_attr('max_level')[0], 
                                '__class__', 
                                14
                            )
                            if hasattr(self.vec_env, 'get_attr'):
                                max_level = self.vec_env.get_attr('max_level')[0]
                            else:
                                max_level = 14
                            
                            if self.current_level < max_level:
                                self.current_level += 1
                                
                                # Update ALL environments to new level
                                self.vec_env.env_method(
                                    'set_level', self.current_level
                                )
                                
                                logger.info(
                                    f"🎓 Curriculum increased to level {self.current_level} "
                                    f"(success rate: {recent_success:.2%} over last "
                                    f"{self.curriculum_window} episodes)"
                                )
                                
                                # Reset tracking for new level
                                self.episode_results = []
                
                # Log current level
                if 'level' in info:
                    self.logger.record("curriculum/current_level", info['level'])
        
        # Always log the callback's current level
        self.logger.record("curriculum/target_level", self.current_level)
        return True


def create_ppo_agent(
    env,
    config: Dict[str, Any],
    tensorboard_log: str,
    use_masking: bool = True,
) -> PPO:
    """
    Create PPO agent.
    
    Args:
        env: Environment or vectorized environment
        config: PPO hyperparameters
        tensorboard_log: Path for tensorboard logs
        use_masking: Whether to use MaskablePPO
    
    Returns:
        PPO or MaskablePPO agent
    """
    policy_kwargs = {}
    if 'net_arch' in config:
        policy_kwargs['net_arch'] = config.get('net_arch')
    
    if use_masking:
        if not MASKABLE_PPO_AVAILABLE:
            raise ImportError(
                "MaskablePPO requires sb3-contrib. "
                "Install with: pip install sb3-contrib"
            )
        
        # Note: MaskablePPO does not support use_sde parameter
        agent = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            learning_rate=config.get('learning_rate', 3e-4),
            n_steps=config.get('n_steps', 2048),
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            ent_coef=config.get('ent_coef', 0.01),
            vf_coef=config.get('vf_coef', 0.5),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs or None,
            verbose=1,
        )
    else:
        agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get('learning_rate', 3e-4),
            n_steps=config.get('n_steps', 2048),
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            ent_coef=config.get('ent_coef', 0.01),
            vf_coef=config.get('vf_coef', 0.5),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            use_sde=config.get('use_sde', False),
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs or None,
            verbose=1,
        )
    
    return agent


def create_a2c_agent(
    env,
    config: Dict[str, Any],
    tensorboard_log: str,
) -> A2C:
    """
    Create A2C agent.
    
    Args:
        env: Environment or vectorized environment
        config: A2C hyperparameters
        tensorboard_log: Path for tensorboard logs
    
    Returns:
        A2C agent
    """
    agent = A2C(
        "MlpPolicy",
        env,
        learning_rate=config.get('learning_rate', 7e-4),
        n_steps=config.get('n_steps', 5),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 1.0),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        use_rms_prop=config.get('use_rms_prop', True),
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    
    return agent


def create_dqn_agent(
    env,
    config: Dict[str, Any],
    tensorboard_log: str,
) -> DQN:
    """
    Create DQN agent.
    
    Args:
        env: Environment (DQN doesn't support vectorized envs)
        config: DQN hyperparameters
        tensorboard_log: Path for tensorboard logs
    
    Returns:
        DQN agent
    """
    agent = DQN(
        "MlpPolicy",
        env,
        learning_rate=config.get('learning_rate', 1e-4),
        buffer_size=config.get('buffer_size', 100000),
        learning_starts=config.get('learning_starts', 10000),
        batch_size=config.get('batch_size', 32),
        tau=config.get('tau', 1.0),
        gamma=config.get('gamma', 0.99),
        train_freq=config.get('train_freq', 4),
        gradient_steps=config.get('gradient_steps', 1),
        target_update_interval=config.get('target_update_interval', 10000),
        exploration_fraction=config.get('exploration_fraction', 0.1),
        exploration_initial_eps=config.get('exploration_initial_eps', 1.0),
        exploration_final_eps=config.get('exploration_final_eps', 0.05),
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    
    return agent


def get_agent_class(algo_name: str):
    """Get agent class by name."""
    algo_name = algo_name.lower()
    
    if algo_name == 'ppo':
        return create_ppo_agent
    elif algo_name == 'a2c':
        return create_a2c_agent
    elif algo_name == 'dqn':
        return create_dqn_agent
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
