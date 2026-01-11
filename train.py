"""
Training script for Ball Sort Puzzle RL agents.

Supports PPO, A2C, DQN with configurable hyperparameters.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import ConfigManager, setup_directories, get_run_name
from agents.agent_factory import (
    make_vec_env,
    get_agent_class,
    CurriculumCallback,
    create_env,
)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent on Ball Sort Puzzle"
    )
    
    parser.add_argument(
        '--algo',
        type=str,
        default='ppo',
        choices=['ppo', 'a2c', 'dqn'],
        help='RL algorithm to use'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Total training timesteps (overrides config)'
    )
    
    parser.add_argument(
        '--n-envs',
        type=int,
        default=None,
        help='Number of parallel environments (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--level',
        type=int,
        default=None,
        help='Initial level (overrides config)'
    )
    
    parser.add_argument(
        '--use-masking',
        action='store_true',
        default=True,
        help='Use action masking (default: True)'
    )
    
    parser.add_argument(
        '--no-masking',
        action='store_true',
        help='Disable action masking'
    )
    
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Custom run name'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    setup_directories(config)
    
    # Override config with command line arguments
    algo = args.algo
    timesteps = args.timesteps or config.get('training.total_timesteps', 100000)
    n_envs = args.n_envs or config.get('training.n_envs', 8)
    seed = args.seed or config.get('training.seed', 42)
    use_masking = not args.no_masking and args.use_masking
    
    # DQN doesn't support vectorized envs
    if algo == 'dqn':
        n_envs = 1
        logger.info("DQN doesn't support vectorized envs, using n_envs=1")
    
    # Environment configuration (curriculum managed by callback, not env)
    env_kwargs = {
        'n_max': config.get('env.n_max', 14),
        'height': config.get('env.height', 4),
        'initial_level': args.level or config.get('env.initial_level', 1),
        'max_level': config.get('env.max_level', 14),
        'max_steps': config.get('env.max_steps', 500),
    }
    
    # Generate run name
    run_name = args.run_name or get_run_name(algo, env_kwargs['initial_level'])
    
    logger.info(f"Starting training: {run_name}")
    logger.info(f"Algorithm: {algo.upper()}")
    logger.info(f"Timesteps: {timesteps:,}")
    logger.info(f"Environments: {n_envs}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Action masking: {use_masking}")
    logger.info(f"Initial level: {env_kwargs['initial_level']}")
    
    # Save configuration
    config.save(run_name)
    
    # Create environment
    use_one_hot = config.get('env.use_one_hot', True)
    
    if n_envs > 1:
        env = make_vec_env(
            n_envs=n_envs,
            env_kwargs=env_kwargs,
            use_masking=use_masking,
            seed=seed,
            use_one_hot=use_one_hot,
        )
    else:
        from stable_baselines3.common.monitor import Monitor
        env_fn = create_env(env_kwargs, use_masking, 0, seed, use_one_hot)
        env = env_fn()
    
    # Get algorithm hyperparameters
    algo_config = config.get(algo, {})
    
    # Tensorboard log path
    tensorboard_log = str(Path(config.get('paths.logs', 'result/logs')) / algo)
    
    # Create agent
    agent_factory = get_agent_class(algo)
    agent = agent_factory(
        env=env,
        config=algo_config,
        tensorboard_log=tensorboard_log,
        use_masking=use_masking if algo == 'ppo' else False,
    )
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_path = Path(config.get('paths.models', 'result/models')) / algo / run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.get('training.save_freq', 100000) // n_envs, 1),
        save_path=str(checkpoint_path),
        name_prefix=algo,
        save_replay_buffer=True if algo == 'dqn' else False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Eval callback with early stop on no improvement
    eval_freq_cfg = config.get('training.eval_freq', 10000)
    if eval_freq_cfg and eval_freq_cfg > 0:
        eval_env = make_vec_env(
            n_envs=1,
            env_kwargs=env_kwargs,
            use_masking=use_masking,
            seed=seed + 1000,
            use_one_hot=use_one_hot,
            vec_env_cls=DummyVecEnv,
        )
        eval_env = VecMonitor(eval_env)
        eval_freq = max(eval_freq_cfg // max(1, n_envs), 1)
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=10,
            verbose=1,
        )
        best_model_path = Path(config.get('paths.models', 'result/models')) / algo / run_name / "best"
        best_model_path.mkdir(parents=True, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_path),
            log_path=str(Path(config.get('paths.logs', 'result/logs')) / algo),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            callback_after_eval=stop_callback,
        )
        callbacks.append(eval_callback)
    
    # Curriculum callback with strict parameters
    curriculum_callback = CurriculumCallback(
        vec_env=env,  # Pass the vectorized environment
        curriculum_window=500,  # Require 500 episodes before advancing
        curriculum_threshold=0.95,  # Must achieve 95% success rate
        verbose=1,
    )
    callbacks.append(curriculum_callback)
    
    # Train
    logger.info("Starting training...")
    try:
        agent.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            log_interval=config.get('training.log_interval', 10),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Save final model
    final_model_path = Path(config.get('paths.models', 'result/models')) / algo / f"{run_name}_final.zip"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(final_model_path))
    
    logger.info(f"Training complete! Model saved to: {final_model_path}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
