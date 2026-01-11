"""
Visual demo with agent playing - Same visual style as level preview.
"""

import argparse
import time
import pygame
import numpy as np
from pathlib import Path
from envs import BallSortEnv
from envs.wrappers import OneHotObservationWrapper

# Colors for balls
COLORS = [
    (0, 0, 0),        # 0 - empty (black)
    (255, 0, 0),      # 1 - red
    (0, 255, 0),      # 2 - green
    (0, 0, 255),      # 3 - blue
    (255, 255, 0),    # 4 - yellow
    (255, 0, 255),    # 5 - magenta
    (0, 255, 255),    # 6 - cyan
    (255, 128, 0),    # 7 - orange
    (128, 0, 255),    # 8 - purple
    (255, 192, 203),  # 9 - pink
    (165, 42, 42),    # 10 - brown
    (128, 128, 128),  # 11 - gray
    (0, 128, 0),      # 12 - dark green
]


def load_model(model_path, algo):
    """Load trained model."""
    if algo == 'ppo':
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(model_path)
    elif algo == 'a2c':
        from sb3_contrib import MaskableA2C
        return MaskableA2C.load(model_path)
    elif algo == 'dqn':
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def _decode_obs(obs):
    """Convert one-hot obs (n_max, height, channels) to integer ids for rendering."""
    if obs is None:
        return obs
    arr = np.asarray(obs)
    if arr.ndim == 3:
        # last channel is 13 for padding (-1)
        idx = arr.argmax(axis=2)
        idx = idx.astype(int)
        idx[idx == 13] = -1
        return idx
    return arr


def draw_state(screen, font, small_font, obs, level, n_tubes, n_colors, step, reward, action_info=""):
    """Draw the current state with nice visuals."""
    obs_render = _decode_obs(obs)
    screen.fill((30, 30, 30))
    
    # Title
    title = font.render(f'Level {level} - {n_tubes} tubes, {n_colors} colors', True, (255, 255, 255))
    screen.blit(title, (350, 15))
    
    # Step and reward info
    info_text = small_font.render(f'Step: {step} | Reward: {reward:.2f} | {action_info}', True, (200, 200, 200))
    screen.blit(info_text, (300, 50))
    
    # Instructions
    instr = small_font.render('Q: Quit | SPACE: Next episode | Arrow keys: Change level', True, (150, 150, 150))
    screen.blit(instr, (280, 460))
    
    # Draw tubes
    tube_width = 50
    tube_height = 180
    ball_radius = 18
    spacing = 65
    start_x = (1000 - n_tubes * spacing) // 2
    start_y = 90
    
    for tube_idx in range(n_tubes):
        x = start_x + tube_idx * spacing
        
        # Draw tube background
        pygame.draw.rect(screen, (50, 50, 50), (x+2, start_y+2, tube_width-4, tube_height-4))
        pygame.draw.rect(screen, (100, 100, 100), (x, start_y, tube_width, tube_height), 2)
        
        # Tube number
        num = small_font.render(str(tube_idx+1), True, (150, 150, 150))
        screen.blit(num, (x + tube_width//2 - 5, start_y + tube_height + 5))
        
        # Draw balls (bottom of tube = index 0, displayed at bottom of screen)
        for ball_idx in range(4):
            color_id = int(obs_render[tube_idx][ball_idx])  # index 0 = bottom, index 3 = top
            # ball_idx 0 = bottom of screen, ball_idx 3 = top of screen
            ball_y = start_y + tube_height - (ball_idx + 1) * 42 + 15
            
            if color_id > 0:
                pygame.draw.circle(screen, COLORS[color_id], (x + tube_width // 2, ball_y), ball_radius)
                pygame.draw.circle(screen, (255, 255, 255), (x + tube_width // 2, ball_y), ball_radius, 2)
            else:
                # Empty slot
                pygame.draw.circle(screen, (40, 40, 40), (x + tube_width // 2, ball_y), ball_radius, 1)
    
    # Color legend
    legend_y = 330
    legend_text = small_font.render('Colors:', True, (255, 255, 255))
    screen.blit(legend_text, (50, legend_y))
    for i in range(1, min(n_colors + 1, 13)):
        cx = 50 + ((i-1) % 6) * 80
        cy = legend_y + 25 + ((i-1) // 6) * 35
        pygame.draw.circle(screen, COLORS[i], (cx, cy), 12)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 12, 1)
    
    pygame.display.flip()


def run_agent_demo(model, algo, levels, fps=3, record_gif=False, output_dir=None):
    """Run agent on levels with visual display."""
    pygame.init()
    screen = pygame.display.set_mode((1000, 500))
    pygame.display.set_caption('Ball Sort Puzzle - Agent Demo')
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    
    use_masking = algo in ['ppo', 'a2c']
    
    current_level_idx = 0
    running = True
    
    while running and current_level_idx < len(levels):
        level = levels[current_level_idx]
        
        # Create environment for this level (use training defaults + one-hot to match trained policy)
        base_env = BallSortEnv(initial_level=level, max_level=level)
        env = OneHotObservationWrapper(base_env)
        print(f"Env config: loop_penalty_first={getattr(base_env, 'loop_penalty_first', None)}, "
              f"loop_penalty={getattr(base_env, 'loop_penalty', None)}, "
              f"loop_grace={getattr(base_env, 'loop_grace', None)}, "
              f"loop_terminate={getattr(base_env, 'loop_terminate', None)}, "
              f"stale_step_limit={getattr(base_env, 'stale_step_limit', None)}")
        obs, info = env.reset()
        
        n_tubes = base_env._get_n_tubes_for_level(level)
        n_colors = base_env._get_n_colors_for_level(level)
        
        episode_reward = 0
        step = 0
        done = False
        frames = []
        
        print(f"\n=== Level {level}/14 ===")
        
        # Initial state
        render_obs = _decode_obs(obs)
        draw_state(screen, font, small_font, render_obs, level, n_tubes, n_colors, step, episode_reward, "Starting...")
        if record_gif:
            frames.append(pygame.surfarray.array3d(screen).transpose([1, 0, 2]))
        
        time.sleep(1.0 / fps)
        
        while not done and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        done = True  # Skip to next episode
                    elif event.key == pygame.K_RIGHT:
                        current_level_idx = min(len(levels) - 1, current_level_idx + 1)
                        done = True
                    elif event.key == pygame.K_LEFT:
                        current_level_idx = max(0, current_level_idx - 1)
                        done = True
            
            if not running or done:
                break
            
            # Get action from model
            if use_masking:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Decode action
            source = int(action) // env.n_max
            dest = int(action) % env.n_max
            action_info = f"Move: tube {source+1} -> tube {dest+1}"
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            if info.get("penalty_loop", 0) != 0 or terminated or truncated:
                print(f"Step {step}: penalty_loop={info.get('penalty_loop')}, "
                      f"terminated={terminated}, truncated={truncated}, "
                      f"stale_steps={info.get('stale_steps')}")
            
            # Draw new state
            status = ""
            if terminated and info.get('is_success', False):
                status = " | SOLVED!"
            elif truncated:
                status = " | Max steps reached"
            
            render_obs = _decode_obs(obs)
            draw_state(screen, font, small_font, render_obs, level, n_tubes, n_colors, step, episode_reward, action_info + status)
            
            if record_gif:
                frames.append(pygame.surfarray.array3d(screen).transpose([1, 0, 2]))
            
            clock.tick(fps)
        
        # Show final state with result
        result = "SUCCESS" if info.get('is_success', False) else "FAILED"
        result_color = (0, 255, 0) if info.get('is_success', False) else (255, 0, 0)
        print(f"Level {level}: {result} in {step} steps, reward: {episode_reward:.2f}")
        
        # Draw final result screen with instructions
        if info.get('is_success', False):
            hint = "SPACE: Next level | R: Replay | Q: Quit"
        else:
            hint = "SPACE/R: Retry | Arrow Right: Skip | Q: Quit"
        
        draw_state(screen, font, small_font, obs, level, n_tubes, n_colors, step, episode_reward, hint)
        
        # Draw big result text
        big_font = pygame.font.Font(None, 72)
        result_text = big_font.render(result, True, result_color)
        text_rect = result_text.get_rect(center=(500, 400))
        pygame.draw.rect(screen, (0, 0, 0), text_rect.inflate(20, 10))
        screen.blit(result_text, text_rect)
        pygame.display.flip()
        
        # Save GIF if requested
        if record_gif and frames and output_dir:
            try:
                import imageio
                gif_path = output_dir / f"level_{level}.gif"
                imageio.mimsave(str(gif_path), frames, fps=fps)
                print(f"Saved GIF: {gif_path}")
            except ImportError:
                print("imageio not installed, skipping GIF")
        
        env.close()
        
        # WAIT for user to press SPACE to continue
        level_passed = info.get('is_success', False)
        waiting_for_next = True
        while waiting_for_next and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_next = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                        waiting_for_next = False
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        waiting_for_next = False
                        if level_passed:
                            current_level_idx += 1  # Go to next level only if SUCCESS
                        # else: replay same level
                    elif event.key == pygame.K_RIGHT:
                        # Force skip to next level (manual override)
                        current_level_idx = min(len(levels) - 1, current_level_idx + 1)
                        waiting_for_next = False
                    elif event.key == pygame.K_LEFT:
                        current_level_idx = max(0, current_level_idx - 1)
                        waiting_for_next = False
                    elif event.key == pygame.K_r:
                        # Replay same level
                        waiting_for_next = False
            time.sleep(0.05)
    
    pygame.quit()
    print("\nDemo finished!")


def main():
    parser = argparse.ArgumentParser(description='Visual Agent Demo')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'a2c', 'dqn'])
    parser.add_argument('--levels', type=str, default='1-14', help='Levels to play (e.g., "1-14" or "1,3,5")')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second (lower = slower to visualize moves)')
    parser.add_argument('--record-gif', action='store_true', help='Record GIFs')
    args = parser.parse_args()
    
    # Parse levels
    if '-' in args.levels:
        start, end = map(int, args.levels.split('-'))
        levels = list(range(start, end + 1))
    else:
        levels = [int(x) for x in args.levels.split(',')]
    
    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.algo)
    
    output_dir = None
    if args.record_gif:
        output_dir = Path("result/episodes")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running demo on levels: {levels}")
    run_agent_demo(model, args.algo, levels, fps=args.fps, record_gif=args.record_gif, output_dir=output_dir)


if __name__ == "__main__":
    main()
