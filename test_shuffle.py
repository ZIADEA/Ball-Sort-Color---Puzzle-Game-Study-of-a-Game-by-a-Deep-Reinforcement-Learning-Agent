"""Test que les puzzles sont bien mélangés."""
from envs import BallSortEnv
import numpy as np

env = BallSortEnv()

print('=== TEST DES PUZZLES MELANGES ===')
print()

for level in [1, 2, 3, 6, 10, 14]:
    env.set_level(level)
    obs, _ = env.reset()
    
    n_tubes = env._get_n_tubes_for_level(level)
    n_colors = env._get_n_colors_for_level(level)
    
    # Count empty tubes
    empty_tubes = sum(1 for t in range(n_tubes) if all(obs[t][i] == 0 for i in range(4)))
    
    solved_count = 0
    print(f'Niveau {level}: {n_tubes} tubes, {n_colors} couleurs, {empty_tubes} tube(s) vide(s)')
    for t in range(n_tubes):
        tube = obs[t]
        if tube[0] != -1:
            colors = [str(c) if c > 0 else '_' for c in tube]
            # Check if solved (all 4 same color, not empty)
            is_solved = all(c == tube[0] for c in tube) and tube[0] > 0
            status = ' <- RESOLU' if is_solved else ''
            if is_solved:
                solved_count += 1
            print(f'  Tube {t+1}: [{colors[0]:>2}, {colors[1]:>2}, {colors[2]:>2}, {colors[3]:>2}]{status}')
    
    print(f'  => {solved_count}/{n_colors} tubes déjà résolus')
    print()
