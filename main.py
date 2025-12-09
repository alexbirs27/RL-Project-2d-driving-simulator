import pygame

from engine import GameEngine


def main():
    """Main entry point for the game."""
    engine = GameEngine(width=1200, height=800, render=True)
    engine.init()

    clock = pygame.time.Clock()
    target_fps = 60

    while engine.running:
        dt = clock.tick(target_fps) / 1000.0

        actions = engine.handle_events()
        engine.step(actions, dt)
        engine.render()

    engine.quit()


if __name__ == "__main__":
    main()
