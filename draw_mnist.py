import pygame
import sys
import numpy as np
from neuralNetwork import NeuralNetwork
from PIL import Image

nn = NeuralNetwork(784, 128, 10, load=True)

# Initialize Pygame
pygame.init()

# Canvas settings
canvas_size = 28
scale_factor = 20  # Increased from 10 to 20 for larger display
display_size = canvas_size * scale_factor
toolbar_height = 60
window_width = display_size
window_height = display_size + toolbar_height

# Set up display
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("MNIST-like Drawer")

# Create canvas array for grayscale values (0-1 float32)
canvas_array = np.zeros((canvas_size, canvas_size), dtype=np.float32)

# Brush settings (adjusted for better drawing experience)
brush_radius = 2.0  # Slightly larger brush
brush_strength = 0.2
drawing = False
last_pos = None

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Toolbar setup
clear_rect = pygame.Rect(10, display_size + 15, 80, 30)
save_rect = pygame.Rect(100, display_size + 15, 80, 30)

# Font for buttons
font = pygame.font.SysFont(None, 24)


def apply_brush(x, y):
    """Apply soft brush effect to the canvas array"""
    for i in range(int(x - brush_radius), int(x + brush_radius) + 1):
        for j in range(int(y - brush_radius), int(y + brush_radius) + 1):
            if 0 <= i < canvas_size and 0 <= j < canvas_size:
                dx = i - x
                dy = j - y
                distance = np.hypot(dx, dy) * 1.3

                if distance <= brush_radius:
                    contribution = (1 - distance / brush_radius) * brush_strength
                    canvas_array[j, i] = min(canvas_array[j, i] + contribution, 1.0)


def draw_line(start, end):
    """Draw smooth line with soft brush effect"""
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    distance = np.hypot(dx, dy)

    if distance == 0:
        apply_brush(x0, y0)
        return

    steps = int(distance * 2)
    step_x = dx / steps
    step_y = dy / steps

    for i in range(steps + 1):
        x = x0 + step_x * i
        y = y0 + step_y * i
        apply_brush(x, y)


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if y < display_size:
                canvas_x = x // scale_factor
                canvas_y = y // scale_factor

                last_pos = (canvas_x, canvas_y)
                drawing = True
                apply_brush(canvas_x, canvas_y)

            else:
                if clear_rect.collidepoint(x, y):
                    canvas_array.fill(0.0)
                elif save_rect.collidepoint(x, y):
                    try:
                        # Process image without rotation/flipping
                        processed_array = (canvas_array.T * 255).astype(np.uint8)
                        img = Image.fromarray(processed_array)

                        img = img.rotate(-90)

                        img = img.transpose(Image.FLIP_LEFT_RIGHT)

                        img.save("img.png")

                        img = np.array(img)

                        # Prepare network input
                        pixel_data = img.flatten().astype(np.float32)

                        # Make prediction
                        prediction = nn.forward(pixel_data)

                        print("Predicted digit:", np.argmax(prediction))

                        print(f"Confidence: {prediction[np.argmax(prediction)]}")

                    except Exception as e:
                        print("Error during prediction:", e)

        elif event.type == pygame.MOUSEMOTION and drawing:
            if last_pos:
                x, y = event.pos
                canvas_x = max(0, min(x // scale_factor, canvas_size - 1))
                canvas_y = max(0, min(y // scale_factor, canvas_size - 1))

                current_pos = (canvas_x, canvas_y)

                draw_line(last_pos, current_pos)

                last_pos = current_pos

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None

    # Render canvas
    gray_scale = (canvas_array.T * 255).astype(np.uint8)
    gray_rgb = np.repeat(gray_scale[:, :, np.newaxis], 3, axis=2)
    canvas_surface = pygame.surfarray.make_surface(gray_rgb)
    scaled_canvas = pygame.transform.scale(canvas_surface, (display_size, display_size))
    screen.blit(scaled_canvas, (0, 0))

    # Draw grid and UI
    for i in range(canvas_size + 1):
        x = i * scale_factor
        pygame.draw.line(screen, GRAY, (x, 0), (x, display_size))
        pygame.draw.line(screen, GRAY, (0, x), (display_size, x))

    pygame.draw.rect(screen, GRAY, (0, display_size, window_width, toolbar_height))
    pygame.draw.rect(screen, WHITE, clear_rect)
    pygame.draw.rect(screen, WHITE, save_rect)
    screen.blit(font.render('Clear', True, BLACK), clear_rect.move(10, 5))
    screen.blit(font.render('Predict', True, BLACK), save_rect.move(10, 5))

    pygame.display.flip()

pygame.quit()
sys.exit()