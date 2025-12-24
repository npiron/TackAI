
import math

# Window & Game
W, H = 1400, 1000
FPS = 50

# Physics constants
MAX_SPEED = 32.0
ACCEL = 0.18        # Reduced for smoother acceleration
BRAKE = 0.5
FRICTION = 0.992
TURN_SPEED = 0.08
LATERAL_GRIP = 0.2  # Increased for better grip (less drift)
COLLISION_SPEED_LOSS = 0.9

# Track config
TRACK_WIDTH = 90
TRACK_OUTER_MARGIN = 40
TRACK_INNER_MARGIN = TRACK_OUTER_MARGIN + TRACK_WIDTH

CHECKPOINT_RADIUS = 100
FINISH_RADIUS = 100

# Fonts
FONT_SIZE = 24

# Colors
COLOR_GRASS = (34, 139, 34)       # Forest Green
COLOR_ROAD = (105, 105, 105)      # Dim Gray
COLOR_BORDER = (255, 255, 255)    # White
COLOR_CHECKPOINT = (255, 215, 0)  # Gold
