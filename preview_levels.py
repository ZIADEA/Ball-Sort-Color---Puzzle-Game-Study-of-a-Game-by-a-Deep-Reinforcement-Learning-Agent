"""
Visualisation interactive des niveaux de Ball Sort Puzzle.

Contrôles:
- Flèche droite: niveau suivant
- Flèche gauche: niveau précédent  
- ESPACE: régénérer le même niveau
- Q ou ESC: quitter
"""

import pygame
import numpy as np
from envs import BallSortEnv
import time

# Couleurs pour les balles (index = color_id)
COLORS = [
    (0, 0, 0),        # 0 - empty (noir)
    (255, 50, 50),    # 1 - rouge
    (50, 255, 50),    # 2 - vert
    (50, 100, 255),   # 3 - bleu
    (255, 255, 50),   # 4 - jaune
    (255, 50, 255),   # 5 - magenta
    (50, 255, 255),   # 6 - cyan
    (255, 150, 50),   # 7 - orange
    (150, 50, 255),   # 8 - violet
    (255, 180, 200),  # 9 - rose
    (180, 80, 50),    # 10 - marron
    (150, 150, 150),  # 11 - gris
    (50, 150, 50),    # 12 - vert foncé
]

def main():
    pygame.init()
    screen = pygame.display.set_mode((1100, 550))
    pygame.display.set_caption('Ball Sort Puzzle - Aperçu des Niveaux')
    font = pygame.font.Font(None, 40)
    small_font = pygame.font.Font(None, 26)
    
    env = BallSortEnv()
    
    def draw_level(obs, level, n_tubes, n_colors):
        screen.fill((25, 25, 35))
        
        # Titre
        title = font.render(f'Niveau {level}', True, (255, 255, 255))
        screen.blit(title, (500, 15))
        
        subtitle = small_font.render(f'{n_tubes} tubes, {n_colors} couleurs', True, (180, 180, 180))
        screen.blit(subtitle, (480, 50))
        
        # Instructions
        instr = small_font.render('← → : changer niveau  |  ESPACE : régénérer  |  Q : quitter', True, (120, 120, 120))
        screen.blit(instr, (320, 520))
        
        # Dimensions des tubes
        tube_width = 55
        tube_height = 200
        ball_radius = 22
        spacing = min(70, (1000 - 100) // n_tubes)
        start_x = (1100 - n_tubes * spacing) // 2
        start_y = 90
        
        for tube_idx in range(n_tubes):
            x = start_x + tube_idx * spacing
            
            # Fond du tube
            pygame.draw.rect(screen, (40, 40, 50), 
                           (x + 2, start_y + 2, tube_width - 4, tube_height - 4))
            # Bordure du tube
            pygame.draw.rect(screen, (80, 80, 100), 
                           (x, start_y, tube_width, tube_height), 3)
            
            # Numéro du tube
            num = small_font.render(str(tube_idx + 1), True, (100, 100, 100))
            screen.blit(num, (x + tube_width // 2 - 6, start_y + tube_height + 8))
            
            # Dessiner les balles (de bas en haut)
            for ball_idx in range(4):
                # obs est stocké [tube][position] où position 0 = bas
                color_id = obs[tube_idx][ball_idx]
                ball_y = start_y + tube_height - (ball_idx + 1) * 48 + 20
                ball_x = x + tube_width // 2
                
                if color_id > 0:
                    # Balle colorée
                    pygame.draw.circle(screen, COLORS[color_id], (ball_x, ball_y), ball_radius)
                    # Effet 3D
                    pygame.draw.circle(screen, (255, 255, 255), (ball_x - 6, ball_y - 6), 6)
                    # Contour
                    pygame.draw.circle(screen, (200, 200, 200), (ball_x, ball_y), ball_radius, 2)
                elif color_id == 0:
                    # Emplacement vide (subtil)
                    pygame.draw.circle(screen, (35, 35, 45), (ball_x, ball_y), ball_radius - 2, 1)
        
        # Légende des couleurs
        legend_y = 330
        legend_text = small_font.render('Couleurs du niveau:', True, (200, 200, 200))
        screen.blit(legend_text, (50, legend_y))
        
        for i in range(1, min(n_colors + 1, 13)):
            row = (i - 1) // 6
            col = (i - 1) % 6
            cx = 70 + col * 90
            cy = legend_y + 35 + row * 50
            
            pygame.draw.circle(screen, COLORS[i], (cx, cy), 18)
            pygame.draw.circle(screen, (255, 255, 255), (cx - 5, cy - 5), 5)
            pygame.draw.circle(screen, (200, 200, 200), (cx, cy), 18, 2)
        
        # Info difficulté
        if level <= 3:
            diff = "Facile"
            diff_color = (100, 255, 100)
        elif level <= 7:
            diff = "Moyen"
            diff_color = (255, 255, 100)
        elif level <= 11:
            diff = "Difficile"
            diff_color = (255, 150, 100)
        else:
            diff = "Expert"
            diff_color = (255, 100, 100)
        
        diff_text = small_font.render(f'Difficulté: {diff}', True, diff_color)
        screen.blit(diff_text, (50, 450))
        
        pygame.display.flip()
    
    current_level = 1
    running = True
    
    print("🎮 Visualisation des niveaux Ball Sort Puzzle")
    print("   ← → : changer niveau")
    print("   ESPACE : régénérer")
    print("   Q : quitter")
    print()
    
    while running:
        env.set_level(current_level)
        obs, info = env.reset()
        n_tubes = env._get_n_tubes_for_level(current_level)
        n_colors = env._get_n_colors_for_level(current_level)
        
        draw_level(obs, current_level, n_tubes, n_colors)
        print(f"Niveau {current_level}: {n_tubes} tubes, {n_colors} couleurs")
        
        waiting = True
        while waiting and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                        waiting = False
                    elif event.key == pygame.K_RIGHT:
                        current_level = min(14, current_level + 1)
                        waiting = False
                    elif event.key == pygame.K_LEFT:
                        current_level = max(1, current_level - 1)
                        waiting = False
                    elif event.key == pygame.K_SPACE:
                        waiting = False  # Régénérer le même niveau
            
            time.sleep(0.03)
    
    pygame.quit()
    print("\n✓ Visualisation terminée!")


if __name__ == "__main__":
    main()
