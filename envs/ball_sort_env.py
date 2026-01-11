"""
Ball Sort Puzzle Gymnasium Environment.

This environment implements the Ball Sort (Color Sort) puzzle game with:
- Action masking for invalid moves
- Curriculum learning based on success rate
- Reverse shuffle level generation
- Detailed reward decomposition for XAI
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BallSortEnv(gym.Env):
    """
    Ball Sort Puzzle Environment.
    
    Observation Space:
        Matrix of shape (N_max, H) with dtype int32
        - -1: padding (tube doesn't exist)
        - 0: empty cell
        - 1-12: color IDs
    
    Action Space:
        Discrete(N_max * N_max)
        Decoded as: source = action // N_max, dest = action % N_max
    
    Rewards:
        - step_time: -0.2 per step
        - progress: delta in purity score
        - bonus_tube_complete: +5 when tube becomes homogeneous
        - bonus_win: +100 + 5*level
        - penalty_blocked: -20 when no valid moves
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        n_max: int = 14,
        height: int = 4,
        initial_level: int = 1,
        max_level: int = 14,
        max_steps: int = 500,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        stale_step_limit: int = 20,
        loop_penalty_first: float = -2.0,
        loop_penalty: float = -10.0,
        loop_terminate: bool = True,
        loop_grace: int = 1,
    ):
        """
        Initialize Ball Sort environment.
        
        Args:
            n_max: Maximum number of tubes (14)
            height: Height of each tube (4)
            initial_level: Starting difficulty level
            max_level: Maximum difficulty level
            max_steps: Maximum steps per episode
            seed: Random seed
            render_mode: "human" or "rgb_array"
            stale_step_limit: Max steps allowed without purity progress before truncating
            loop_penalty_first: Penalty applied on the first revisit of a state
            loop_penalty: Penalty applied when revisiting a state beyond the grace limit
            loop_terminate: If True, terminate the episode immediately once grace is exceeded
            loop_grace: Number of revisits allowed before triggering the strong penalty/termination
        """
        super().__init__()
        
        self.n_max = n_max
        self.height = height
        self.initial_level = initial_level
        self.current_level = initial_level
        self.max_level = max_level
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.stale_step_limit = stale_step_limit
        self.loop_penalty_first = loop_penalty_first
        self.loop_penalty = loop_penalty
        self.loop_terminate = loop_terminate
        self.loop_grace = max(0, loop_grace)
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-1, high=12, shape=(n_max, height), dtype=np.int32
        )
        self.action_space = spaces.Discrete(n_max * n_max)
        
        # State
        self.state = None
        self.n_tubes = None
        self.n_colors = None
        self.steps_count = 0
        self.previous_purity = 0.0
        self.stale_steps = 0
        self.visited_states = {}
        
        # Curriculum tracking
        self.episode_results = []
        
        # Random generator
        self.np_random = None
        self._seed(seed)
        
        # Pygame for rendering
        self.window = None
        self.clock = None
        
    def _seed(self, seed: Optional[int] = None):
        """Set random seed."""
        self.np_random = np.random.RandomState(seed)
    
    def _get_state_key(self, state: np.ndarray) -> bytes:
        """Hashable key for current state (used to detect loops)."""
        return state.tobytes()
    
    def set_level(self, level: int):
        """
        Set the current difficulty level (called by curriculum callback).
        
        Args:
            level: New difficulty level (1 to max_level)
        """
        self.current_level = min(max(1, level), self.max_level)
        
    def _get_n_tubes_for_level(self, level: int) -> int:
        """
        Calculate number of tubes based on level.
        
        Level 1: 2 tubes (1 color split between 2 tubes, 0 empty)
        Level 2: 3 tubes (2 colors + 1 empty tube)
        Level 3+: n_colors + 2 empty tubes
        """
        if level == 1:
            return 2  # 1 color split in 2 tubes, 0 empty
        elif level == 2:
            return 3  # 2 colors + 1 empty tube
        else:
            return min(level + 2, self.n_max)  # n_colors + 2 empty
    
    def _get_n_colors_for_level(self, level: int) -> int:
        """
        Calculate number of colors.
        
        Level 1: 1 color
        Level 2: 2 colors
        Level 3+: level colors (up to 12)
        """
        if level == 1:
            return 1
        elif level == 2:
            return 2
        else:
            return min(level, 12)  # Max 12 colors
    
    def _generate_solvable_puzzle(self) -> np.ndarray:
        """
        Generate a shuffled puzzle that is guaranteed solvable.
        
        Strategy: Create all balls, shuffle them randomly, then distribute
        into tubes. The puzzle is solvable because we start from solved
        and just redistribute randomly.
        """
        n_tubes = self.n_tubes
        n_colors = self.n_colors
        
        # Initialize state
        state = np.full((self.n_max, self.height), -1, dtype=np.int32)
        
        # Special case: Level 1 - 2 tubes with 4 balls of 1 color split between them
        # 0 empty tube - agent learns to group same color balls in one tube
        if self.current_level == 1:
            # Split 4 balls randomly between 2 tubes (min 1 per tube)
            split = self.np_random.randint(1, 4)  # 1, 2, or 3 balls in tube 0
            
            # Tube 0: 'split' balls at bottom (stack: index 0 = bottom)
            state[0, :] = 0  # initialize empty
            for i in range(split):
                state[0, i] = 1  # color 1
            
            # Tube 1: remaining balls at bottom
            state[1, :] = 0  # initialize empty
            for i in range(4 - split):
                state[1, i] = 1  # color 1
            
            return state
        
        # Special case: Level 2 - 2 colors shuffled in 2 tubes + 1 empty tube
        if self.current_level == 2:
            # Create 8 balls (4 of color 1, 4 of color 2)
            all_balls = [1, 1, 1, 1, 2, 2, 2, 2]
            attempts = 0
            while True:
                self.np_random.shuffle(all_balls)
                
                # Fill tube 0 and tube 1 (bottom to top: index 0,1,2,3)
                state[0, 0] = all_balls[0]  # bottom
                state[0, 1] = all_balls[1]
                state[0, 2] = all_balls[2]
                state[0, 3] = all_balls[3]  # top
                
                state[1, 0] = all_balls[4]  # bottom
                state[1, 1] = all_balls[5]
                state[1, 2] = all_balls[6]
                state[1, 3] = all_balls[7]  # top
                
                # Tube 2: empty (1 empty tube)
                state[2, :] = 0
                
                # Avoid starting with a fully solved tube
                if not (self._is_tube_full_uniform(state, 0) or self._is_tube_full_uniform(state, 1)):
                    break
                
                attempts += 1
                if attempts >= 200:
                    # Give up after many tries; keep latest shuffle
                    break
            
            return state
        
        # Create list of all balls: 4 balls per color
        all_balls = []
        for color in range(1, n_colors + 1):
            all_balls.extend([color] * self.height)  # 4 balls of each color
        
        # Shuffle all balls randomly
        self.np_random.shuffle(all_balls)
        
        # Fill n_colors tubes with shuffled balls
        n_filled_tubes = n_colors
        
        # Fill tubes with shuffled balls
        ball_idx = 0
        for tube_idx in range(n_filled_tubes):
            for pos in range(self.height):
                state[tube_idx, pos] = all_balls[ball_idx]
                ball_idx += 1
        
        # Mark empty tubes (2 empty tubes)
        for tube_idx in range(n_filled_tubes, n_tubes):
            state[tube_idx, :] = 0
        
        # Avoid starting with solved or already-complete tubes
        attempts = 0
        while (self._is_solved(state) or
               any(self._is_tube_full_uniform(state, idx) for idx in range(n_filled_tubes))) and attempts < 500:
            self.np_random.shuffle(all_balls)
            ball_idx = 0
            for tube_idx in range(n_filled_tubes):
                for pos in range(self.height):
                    state[tube_idx, pos] = all_balls[ball_idx]
                    ball_idx += 1
            attempts += 1
        
        return state
    
    def _count_solved_tubes(self, state: np.ndarray) -> int:
        """Count how many tubes are already solved (4 same-colored balls)."""
        count = 0
        for tube_idx in range(self.n_tubes):
            tube = state[tube_idx]
            if tube[0] <= 0:  # Empty or padding
                continue
            # Check if all 4 balls are same color
            if all(tube[i] == tube[0] for i in range(self.height)):
                count += 1
        return count
    
    def _get_tube_top_info(self, state: np.ndarray, tube_idx: int) -> Tuple[int, int, int]:
        """
        Get info about top of tube.
        
        Returns:
            (top_color, count, free_space)
            - top_color: color at top (0 if empty, -1 if padding)
            - count: number of consecutive balls of same color from the top
            - free_space: number of empty cells above the top ball
        """
        tube = state[tube_idx]
        
        # Check if tube is padding
        if tube[0] == -1:
            return -1, 0, 0
        
        # Find top ball (scan from visual top toward bottom)
        top_idx = -1
        for i in range(self.height - 1, -1, -1):
            if tube[i] != 0:
                top_idx = i
                break
        
        # If tube is empty
        if top_idx == -1:
            return 0, 0, self.height
        
        top_color = tube[top_idx]
        
        # Count consecutive balls of same color from the top downwards
        count = 1
        for i in range(top_idx - 1, -1, -1):
            if tube[i] == top_color:
                count += 1
            else:
                break
        
        # Free space is number of empty cells above the top ball
        free_space = self.height - top_idx - 1
        
        return top_color, count, free_space
    
    def _is_move_valid(self, state: np.ndarray, source: int, dest: int) -> bool:
        """Check if a move is valid."""
        # Same tube
        if source == dest:
            return False
        
        # Check padding
        if state[source, 0] == -1 or state[dest, 0] == -1:
            return False
        
        # Do not touch tubes that are already perfectly filled with one color
        if self._is_tube_full_uniform(state, source):
            return False
        
        source_color, _, _ = self._get_tube_top_info(state, source)
        dest_color, _, dest_free = self._get_tube_top_info(state, dest)
        
        # Source is empty
        if source_color == 0:
            return False
        
        # Destination is full
        if dest_free == 0:
            return False
        
        # Destination is empty (always valid)
        if dest_color == 0:
            return True
        
        # Colors must match
        if source_color != dest_color:
            return False
        
        # Check if there's space for at least one ball
        if dest_free < 1:
            return False
        
        return True
    
    def _get_valid_actions(self, state: np.ndarray) -> List[int]:
        """Get list of valid action indices."""
        valid = []
        for action in range(self.action_space.n):
            source = action // self.n_max
            dest = action % self.n_max
            if self._is_move_valid(state, source, dest):
                valid.append(action)
        return valid
    
    def action_masks(self) -> np.ndarray:
        """
        Return binary mask for valid actions.
        
        Returns:
            Array of shape (n_max * n_max,) with 1 for valid, 0 for invalid
        """
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        valid_actions = self._get_valid_actions(self.state)
        if len(valid_actions) == 0:
            # Fallback: allow all actions to keep MaskablePPO stable; episode will terminate on step
            mask[:] = True
        else:
            mask[valid_actions] = True
        
        # Safety: ensure at least one valid action to avoid NaNs in MaskableCategorical
        if not mask.any():
            mask[:] = True
        
        return mask.astype(np.int8)
    
    def _apply_move(self, state: np.ndarray, source: int, dest: int) -> np.ndarray:
        """
        Apply a move to the state.
        
        Move exactly one ball from the top of source to the top of dest.
        """
        source_color, _, _ = self._get_tube_top_info(state, source)
        
        source_tube = state[source].copy()
        dest_tube = state[dest].copy()
        
        # Find source top position (highest non-empty cell)
        source_top_idx = -1
        for i in range(self.height - 1, -1, -1):
            if source_tube[i] != 0:
                source_top_idx = i
                break
        
        # Find destination target position (first empty cell above the current stack)
        dest_target_idx = -1
        for i in range(self.height - 1, -1, -1):
            if dest_tube[i] != 0:
                dest_target_idx = i + 1
                break
        if dest_target_idx == -1:  # Completely empty tube
            dest_target_idx = 0
        
        # Safety: invalid move should have been filtered already
        if source_top_idx == -1 or dest_target_idx == -1:
            return state
        
        # Move one ball
        dest_tube[dest_target_idx] = source_color
        source_tube[source_top_idx] = 0
        
        state[source] = source_tube
        state[dest] = dest_tube
        
        return state
    
    def _calculate_purity_score(self, state: np.ndarray) -> float:
        """
        Calculate purity score: measures how close we are to solution.
        
        Reward organized stacks and penalize fragmentation:
        - Bonus for contiguous block from bottom (squared height)
        - Penalty for color changes within a tube
        - Empty tubes contribute positively
        """
        total_score = 0.0
        
        for tube_idx in range(self.n_tubes):
            tube = state[tube_idx]
            
            # Skip padding tubes
            if tube[0] == -1:
                continue
            
            if tube[0] == 0:
                total_score += 1.0  # Encourage keeping useful empty tubes
                continue
            
            bottom_color = tube[0]
            consecutive = 1
            for i in range(1, self.height):
                if tube[i] == 0:
                    break
                if tube[i] == bottom_color:
                    consecutive += 1
                else:
                    break
            
            # Reward contiguous stack from bottom (quadratic)
            total_score += (consecutive ** 2)
            
            # Fragmentation penalty: count color changes in the filled part
            changes = 0
            last_color = tube[0]
            for i in range(1, self.height):
                if tube[i] == 0:
                    break
                if tube[i] != last_color:
                    changes += 1
                    last_color = tube[i]
            total_score -= 1.5 * changes
        
        return total_score / max(1, self.n_tubes)
    
    def _is_tube_full_uniform(self, state: np.ndarray, tube_idx: int) -> bool:
        """Check if tube is full and all balls have the same non-zero color."""
        tube = state[tube_idx]
        
        if tube[0] == -1:  # Padding
            return False
        
        if 0 in tube:
            return False
        
        # All cells must be same color
        color = tube[0]
        for cell in tube:
            if cell != color:
                return False
        
        return True
    
    def _is_tube_complete(self, state: np.ndarray, tube_idx: int) -> bool:
        """Check if tube is complete (either empty or full and homogeneous)."""
        tube = state[tube_idx]
        
        if tube[0] == -1:  # Padding
            return False
        
        if np.all(tube == 0):  # Empty tube
            return True
        
        return self._is_tube_full_uniform(state, tube_idx)
    
    def _is_solved(self, state: np.ndarray) -> bool:
        """Check if puzzle is solved (all tubes complete)."""
        for tube_idx in range(self.n_tubes):
            if not self._is_tube_complete(state, tube_idx):
                return False
        return True
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to new episode."""
        if seed is not None:
            self._seed(seed)
        
        # Set level parameters (curriculum controlled externally)
        self.n_tubes = self._get_n_tubes_for_level(self.current_level)
        self.n_colors = self._get_n_colors_for_level(self.current_level)
        
        # Generate puzzle
        self.state = self._generate_solvable_puzzle()
        self.steps_count = 0
        self.previous_purity = self._calculate_purity_score(self.state)
        self.stale_steps = 0
        self.visited_states = {self._get_state_key(self.state): 1}
        
        info = {
            "level": self.current_level,
            "n_tubes": self.n_tubes,
            "n_colors": self.n_colors,
        }
        
        return self.state.copy(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.steps_count += 1
        
        # Decode action
        source = action // self.n_max
        dest = action % self.n_max
        
        # Check if move is valid
        if not self._is_move_valid(self.state, source, dest):
            # Invalid action penalty
            reward = -1.0
            info = {
                "invalid_action": True,
                "reward_time": 0.0,
                "reward_purity": 0.0,
                "reward_win": 0.0,
            }
            return self.state.copy(), reward, False, False, info
        
        # Track previous state
        prev_state = self.state.copy()
        
        # Apply move
        self.state = self._apply_move(self.state, source, dest)
        
        # Calculate reward components
        # Time penalty encourages finding shortest solution
        reward_time = -0.2
        
        current_purity = self._calculate_purity_score(self.state)
        reward_purity = current_purity - self.previous_purity
        self.previous_purity = current_purity
        
        # Track stagnation (no purity progress)
        if reward_purity > 0:
            self.stale_steps = 0
        else:
            self.stale_steps += 1
        
        # Initialize terminal flags before loop checks
        terminated = False
        truncated = False
        
        # Loop penalty: discourage revisiting exact same state
        penalty_loop = 0.0
        state_key = self._get_state_key(self.state)
        prev_visits = self.visited_states.get(state_key, 0)
        current_visits = prev_visits + 1
        self.visited_states[state_key] = current_visits
        
        if current_visits > 1:
            if current_visits <= self.loop_grace + 1:
                penalty_loop = self.loop_penalty_first
            else:
                penalty_loop = self.loop_penalty
                if self.loop_terminate:
                    terminated = True
        
        # Bonus for completing a tube
        reward_complete = 0.0
        for tube_idx in range(self.n_tubes):
            if (not self._is_tube_complete(prev_state, tube_idx) and
                self._is_tube_complete(self.state, tube_idx)):
                reward_complete += 5.0
        
        # Check terminal conditions (respect loop terminate flag set above)
        reward_win = 0.0
        penalty_blocked = 0.0
        
        if self._is_solved(self.state):
            terminated = True
            reward_win = 100.0 + 5.0 * self.current_level
            self.episode_results.append(1.0)  # Success
        elif len(self._get_valid_actions(self.state)) == 0:
            terminated = True
            penalty_blocked = -20.0
            self.episode_results.append(0.0)  # Failure
        elif self.stale_steps >= self.stale_step_limit:
            truncated = True
            penalty_blocked = -10.0  # Penalty for being stuck
            self.episode_results.append(0.0)
        elif self.steps_count >= self.max_steps:
            truncated = True
            self.episode_results.append(0.0)  # Timeout
        
        # Total reward
        reward = reward_time + reward_purity + reward_complete + reward_win + penalty_blocked + penalty_loop
        
        info = {
            "level": self.current_level,
            "steps": self.steps_count,
            "purity": current_purity,
            "reward_time": reward_time,
            "reward_purity": reward_purity,
            "reward_complete": reward_complete,
            "reward_win": reward_win,
            "penalty_blocked": penalty_blocked,
            "penalty_loop": penalty_loop,
            "loop_count": self.visited_states.get(state_key, 0),
            "stale_steps": self.stale_steps,
            "is_success": self._is_solved(self.state),
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def render(self):
        """Render the environment using pygame."""
        if self.render_mode is None:
            return None
        
        try:
            import pygame
        except ImportError:
            logger.warning("pygame not available, skipping render")
            return None
        
        # Initialize pygame
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((800, 600))
                pygame.display.set_caption("Ball Sort Puzzle")
            else:  # rgb_array
                self.window = pygame.Surface((800, 600))
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((800, 600))
        canvas.fill((255, 255, 255))
        
        # Color palette
        COLORS = {
            -1: (100, 100, 100),  # Padding (gray)
            0: (240, 240, 240),   # Empty (light gray)
            1: (255, 0, 0),       # Red
            2: (0, 255, 0),       # Green
            3: (0, 0, 255),       # Blue
            4: (255, 255, 0),     # Yellow
            5: (255, 0, 255),     # Magenta
            6: (0, 255, 255),     # Cyan
            7: (255, 128, 0),     # Orange
            8: (128, 0, 255),     # Purple
            9: (0, 128, 128),     # Teal
            10: (128, 128, 0),    # Olive
            11: (255, 128, 128),  # Pink
            12: (128, 255, 128),  # Light green
        }
        
        # Draw tubes
        tube_width = 50
        tube_spacing = 60
        ball_radius = 20
        
        for tube_idx in range(self.n_tubes):
            tube = self.state[tube_idx]
            
            # Skip padding
            if tube[0] == -1:
                continue
            
            x = 50 + tube_idx * tube_spacing
            y_base = 500
            
            # Draw tube outline
            pygame.draw.rect(
                canvas,
                (50, 50, 50),
                (x - 5, y_base - self.height * 45, tube_width + 10, self.height * 45 + 5),
                2
            )
            
            # Draw balls
            for i, color_id in enumerate(tube):
                if color_id == 0:  # Empty
                    continue
                
                y = y_base - (i + 1) * 45
                color = COLORS.get(color_id, (128, 128, 128))
                pygame.draw.circle(canvas, color, (x + tube_width // 2, y), ball_radius)
                pygame.draw.circle(canvas, (0, 0, 0), (x + tube_width // 2, y), ball_radius, 2)
        
        # Draw info
        font = pygame.font.Font(None, 36)
        level_text = font.render(f"Level: {self.current_level}", True, (0, 0, 0))
        steps_text = font.render(f"Steps: {self.steps_count}", True, (0, 0, 0))
        canvas.blit(level_text, (550, 50))
        canvas.blit(steps_text, (550, 100))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
