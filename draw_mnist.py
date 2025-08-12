import pygame
import sys
import numpy as np
from neuralNetwork import NeuralNetwork
from PIL import Image
import time

# ------------------------
# Config
# ------------------------
CANVAS_PIXELS = 28
SCALE = 20  # size multiplier for each canvas pixel
DISPLAY_SIZE = CANVAS_PIXELS * SCALE
SIDEBAR_WIDTH = 320  # room for predictions / controls
TOOLBAR_HEIGHT = 24
WINDOW_WIDTH = DISPLAY_SIZE + SIDEBAR_WIDTH
WINDOW_HEIGHT = DISPLAY_SIZE + TOOLBAR_HEIGHT
FPS = 60
AUTO_PREDICT_INTERVAL = 0.12  # seconds

# ------------------------
# Initialize network
# ------------------------
nn = NeuralNetwork(784, 1024, 10, load=True)

# ------------------------
# Pygame init
# ------------------------
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Test my neural network!")
clock = pygame.time.Clock()

# Fonts
TITLE_FONT = pygame.font.SysFont("Arial", 22, bold=True)
UI_FONT = pygame.font.SysFont("Arial", 16)
SMALL_FONT = pygame.font.SysFont("Arial", 14)

# ------------------------
# Colors
# ------------------------
BG = (24, 26, 31)  # dark background
PANEL = (30, 33, 38)
CANVAS_BG = (12, 12, 14)
GRID = (40, 42, 48)
ACCENT = (100, 180, 255)
WHITE = (255, 255, 255)
LIGHT = (200, 200, 200)
BUTTON = (240, 240, 240)
BUTTON_TEXT = (20, 20, 20)

# ------------------------
# Canvas data
# ------------------------
canvas = np.zeros((CANVAS_PIXELS, CANVAS_PIXELS), dtype=np.float32)  # 0..1
brush_radius = 2.4
brush_strength = 0.26
is_drawing = False
last_pos = None

clear_btn_rect = pygame.Rect(DISPLAY_SIZE + 20, 54, SIDEBAR_WIDTH - 40, 38)
save_btn_rect = pygame.Rect(DISPLAY_SIZE + 20, 100, SIDEBAR_WIDTH - 40, 32)

last_forward_time = 0
predictions = np.zeros(10, dtype=np.float32)
auto_predict = True
show_grid = False

def apply_brush(x, y, radius=None, strength=None):
    """Apply a soft circular brush to the normalized canvas array.
    x,y are in canvas pixel coordinates (0..CANVAS_PIXELS-1)
    """
    r = radius if radius is not None else brush_radius
    s = strength if strength is not None else brush_strength

    x0 = int(np.floor(x - r))
    x1 = int(np.ceil(x + r))
    y0 = int(np.floor(y - r))
    y1 = int(np.ceil(y + r))

    for i in range(x0, x1 + 1):
        if i < 0 or i >= CANVAS_PIXELS:
            continue
        for j in range(y0, y1 + 1):
            if j < 0 or j >= CANVAS_PIXELS:
                continue
            dx = i - x
            dy = j - y
            dist = np.hypot(dx, dy)
            if dist <= r:
                contrib = (1 - (dist / r)) * s
                canvas[j, i] = min(1.0, canvas[j, i] + contrib)


def draw_line(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    if dist == 0:
        apply_brush(x0, y0)
        return
    steps = max(int(dist * 2), 1)
    sx = dx / steps
    sy = dy / steps
    for i in range(steps + 1):
        apply_brush(x0 + sx * i, y0 + sy * i)


def canvas_to_image_array():
    arr = (canvas.T * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.rotate(-90)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(img)


def predict_once():
    global predictions

    try:
        arr = canvas_to_image_array()
        img_flat = arr.flatten().astype(np.float32)
        out = nn.forward(img_flat)
        out = np.asarray(out, dtype=np.float32)
        if out.sum() > 0:
            probs = out / out.sum()
        else:
            probs = out

        predictions = probs.reshape(-1,)

    except Exception as e:
        print("Prediction error:", e)

def draw_sidebar():
    # sidebar background
    pygame.draw.rect(screen, PANEL, (DISPLAY_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))

    # Title at the top (safe space)
    title = TITLE_FONT.render("MNIST Painter", True, WHITE)
    screen.blit(title, (DISPLAY_SIZE + 20, 14))

    # Clear button (moved down)
    pygame.draw.rect(screen, BUTTON, clear_btn_rect, border_radius=8)
    screen.blit(UI_FONT.render("Clear (C)", True, BUTTON_TEXT), (clear_btn_rect.x + 12, clear_btn_rect.y + 8))

    # Save button (exports 28x28 PNG)
    pygame.draw.rect(screen, BUTTON, save_btn_rect, border_radius=8)
    screen.blit(SMALL_FONT.render("Save 28x28 PNG", True, BUTTON_TEXT), (save_btn_rect.x + 12, save_btn_rect.y + 6))

    # brush size (below buttons)
    bs_text = SMALL_FONT.render(f"Brush: {brush_radius:.1f} px — Mouse wheel to change", True, LIGHT)
    screen.blit(bs_text, (DISPLAY_SIZE + 20, save_btn_rect.y + 44))

    # Auto predict toggle
    ap_text = SMALL_FONT.render(f"Auto-predict: {'ON' if auto_predict else 'OFF'} (Space)", True, LIGHT)
    screen.blit(ap_text, (DISPLAY_SIZE + 20, save_btn_rect.y + 70))

    # Prediction header (moved farther down)
    pred_title = UI_FONT.render("Predictions", True, WHITE)
    pred_title_y = save_btn_rect.y + 110
    screen.blit(pred_title, (DISPLAY_SIZE + 20, pred_title_y))

    # Bars
    bar_x = DISPLAY_SIZE + 20
    bar_y = pred_title_y + 30
    bar_w = SIDEBAR_WIDTH - 60
    max_bar_w = bar_w

    top_digit = int(np.argmax(predictions)) if predictions.sum() > 0 else None

    for i in range(10):
        y = bar_y + i * 26
        # label
        label = UI_FONT.render(str(i), True, WHITE if i == top_digit else LIGHT)
        screen.blit(label, (bar_x, y))
        # background bar
        pygame.draw.rect(screen, (36, 38, 44), (bar_x + 28, y + 4, max_bar_w, 16), border_radius=6)
        # filled bar
        fill_w = int(predictions[i] * max_bar_w)
        if fill_w > 0:
            pygame.draw.rect(screen, ACCENT, (bar_x + 28, y + 4, fill_w, 16), border_radius=6)
        # percent text
        pct = f"{predictions[i]*100:5.1f}%"
        pct_surf = SMALL_FONT.render(pct, True, WHITE)
        screen.blit(pct_surf, (bar_x + 28 + max_bar_w + 8, y))

    if top_digit is not None:
        big = TITLE_FONT.render(f"Top: {top_digit}", True, ACCENT)
        screen.blit(big, (DISPLAY_SIZE + 20, WINDOW_HEIGHT - 85))
        conf = SMALL_FONT.render(f"Confidence: {predictions[top_digit]*100:4.1f}%", True, LIGHT)
        screen.blit(conf, (DISPLAY_SIZE + 20, WINDOW_HEIGHT - 58))

    # small preview of the 28x28 image
    preview = canvas_to_image_array()
    preview_surf = pygame.surfarray.make_surface(np.repeat(preview[:, :, np.newaxis], 3, axis=2))
    psize = 112
    preview_surf = pygame.transform.scale(preview_surf, (psize, psize))
    preview_x = DISPLAY_SIZE + SIDEBAR_WIDTH - 20 - psize
    preview_y = WINDOW_HEIGHT - 20 - psize
    #screen.blit(SMALL_FONT.render("Preview (28x28)", True, LIGHT), (preview_x, preview_y - 18))
    #screen.blit(preview_surf, (preview_x, preview_y))


# Main loop
running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    now = time.time()

    # Auto predict
    if auto_predict and (now - last_forward_time) > AUTO_PREDICT_INTERVAL:
        predict_once()
        last_forward_time = now

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            # only start drawing if inside canvas area (inset 8 px)
            if 8 <= mx < 8 + (DISPLAY_SIZE - 16) and 8 <= my < 8 + (DISPLAY_SIZE - 16):
                cx = (mx - 8) // SCALE
                cy = (my - 8) // SCALE
                is_drawing = True
                last_pos = (cx, cy)
                apply_brush(cx, cy)

            else:
                # clicks on sidebar
                if clear_btn_rect.collidepoint(event.pos):
                    canvas.fill(0.0)
                elif save_btn_rect.collidepoint(event.pos):
                    arr = canvas_to_image_array()
                    Image.fromarray(arr).save("mnist_export.png")
                    print("Saved mnist_export.png")

        elif event.type == pygame.MOUSEMOTION and is_drawing:
            mx, my = event.pos
            cx = max(0, min((mx - 8) // SCALE, CANVAS_PIXELS - 1))
            cy = max(0, min((my - 8) // SCALE, CANVAS_PIXELS - 1))
            cur = (cx, cy)
            if last_pos is not None:
                draw_line(last_pos, cur)
            last_pos = cur

        elif event.type == pygame.MOUSEBUTTONUP:
            is_drawing = False
            last_pos = None
            # immediate predict on mouse up
            predict_once()
            last_forward_time = now

        elif event.type == pygame.MOUSEWHEEL:
            # change brush size
            if event.y > 0:
                brush_radius = min(10.0, brush_radius + 0.4)
            else:
                brush_radius = max(0.8, brush_radius - 0.4)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                canvas.fill(0.0)
            elif event.key == pygame.K_SPACE:
                auto_predict = not auto_predict
            elif event.key == pygame.K_g:
                show_grid = not show_grid
            elif event.key == pygame.K_s:
                arr = canvas_to_image_array()
                Image.fromarray(arr).save("mnist_export.png")
                print("Saved mnist_export.png")

    # Render background
    screen.fill(BG)

    # Draw canvas background BEFORE blitting the pixel surface so it remains visible
    canvas_inset_rect = pygame.Rect(8, 8, DISPLAY_SIZE - 16, DISPLAY_SIZE - 16)
    pygame.draw.rect(screen, CANVAS_BG, canvas_inset_rect, border_radius=10)

    # Render canvas to surface (28x28 -> scaled)
    gray = (canvas.T * 255).astype(np.uint8)
    gray_rgb = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    surf = pygame.surfarray.make_surface(gray_rgb)
    surf = pygame.transform.scale(surf, (DISPLAY_SIZE - 16, DISPLAY_SIZE - 16))
    screen.blit(surf, (8, 8))

    # optional grid
    if show_grid:
        for i in range(CANVAS_PIXELS + 1):
            x = 8 + i * SCALE
            pygame.draw.line(screen, GRID, (x, 8), (x, 8 + DISPLAY_SIZE - 16))
            pygame.draw.line(screen, GRID, (8, 8 + i * SCALE), (8 + DISPLAY_SIZE - 16, 8 + i * SCALE))

    # draw a subtle frame around canvas
    pygame.draw.rect(screen, GRID, canvas_inset_rect, width=2, border_radius=10)

    # Sidebar / controls (draw after canvas to overlay text)
    draw_sidebar()

    pygame.display.flip()

pygame.quit()
sys.exit()
