# Fișier: src/play_manual.py
import sys
import os
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.game.engine import GameEngine

def main():
    # Inițializăm motorul cu randare activată
    engine = GameEngine(width=1200, height=800, render=True)
    engine.init()

    clock = pygame.time.Clock()
    target_fps = 60


    while engine.running:
        # Calculăm timpul dintre frame-uri (Delta Time)
        dt = clock.tick(target_fps) / 1000.0

        # Aici e secretul: handle_events() citește tastatura ta
        # și returnează o listă de acțiuni (ex: [ACCELERATE, TURN_LEFT])
        actions = engine.handle_events()

        # Trimitem acțiunile tale către motorul fizic
        state = engine.step(actions, dt)
        
        # Desenăm pe ecran
        engine.render()

        # Opțional: Afișăm viteza în consolă
        # print(f"Viteză: {state.velocity:.1f}", end='\r')

    engine.quit()

if __name__ == "__main__":
    main()