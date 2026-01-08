"""
Test the F1Tenth track loader with visualization.
"""
import pygame
from f1tenth_track import F1TenthTrack
from car import Car
from renderer import Renderer

def main():
    # Load Spielberg track from CSV centerline
    track = F1TenthTrack(
        csv_path="tracks/Spielberg/Spielberg_centerline.csv",
        scale=50.0,       # pixels per meter
        road_width=100.0  # track width in pixels
    )

    # Create car at start position (scaled for zoom)
    car = Car(*track.start_position, track.start_angle)
    car.width = 20    # Width (side to side)
    car.length = 14  # Length (front to back) - should be visible

    # Create renderer with zoom
    renderer = Renderer(width=1600, height=1200)
    pygame.init()
    renderer.init()

    # Zoom level (higher = more zoomed in)
    zoom_level = 1 # 2.5x zoom

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Handle keyboard input
        from car import Action
        actions = []
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            actions.append(Action.ACCELERATE)
        if keys[pygame.K_DOWN]:
            actions.append(Action.BRAKE)
        if keys[pygame.K_LEFT]:
            actions.append(Action.TURN_LEFT)
        if keys[pygame.K_RIGHT]:
            actions.append(Action.TURN_RIGHT)

        # Update car (dt first, then actions)
        car.update(1.0/60.0, actions)

        # Check if on road
        corners = car.get_corners()
        on_road = all(track.is_on_road(x, y) for x, y in corners)
        car.on_road = on_road

        # Update camera to follow car with zoom
        renderer.camera_x = car.x - (renderer.width / zoom_level) // 2
        renderer.camera_y = car.y - (renderer.height / zoom_level) // 2
        renderer.zoom = zoom_level  # Store zoom for rendering

        # Render
        renderer.render(
            car=car,
            track=track,
            lap_time=0.0,
            lap_complete=False,
            best_time=None
        )

    pygame.quit()

if __name__ == "__main__":
    main()
