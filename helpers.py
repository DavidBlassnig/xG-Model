import math

GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
GOAL_POST_LEFT_Y = 36.0
GOAL_POST_RIGHT_Y = 44.0


def distance_to_goal(x: float, y: float) -> float:
    """Euklidische Distanz vom Schussort zur Tormitte."""
    return math.sqrt((GOAL_X - x) ** 2 + (GOAL_CENTER_Y - y) ** 2)


def angle_to_goal(x: float, y: float) -> float:
    """
    Schusswinkel in Grad (0° = keine Chance, 90° = direkt vor Tor).
    Berechnet den Winkel des sichtbaren Torpfostensegments vom Schussort aus.
    """
    return math.degrees(angle_to_goal_rad(x, y))


def angle_to_goal_rad(x: float, y: float) -> float:
    """
    Schusswinkel in Bogenmass.
    Berechnet über den Winkel zwischen den Vektoren zu beiden Torpfosten.
    """
    if x >= GOAL_X:
        return math.pi / 2

    dx = GOAL_X - x
    dy_left = GOAL_POST_LEFT_Y - y
    dy_right = GOAL_POST_RIGHT_Y - y

    angle_left = math.atan2(dy_left, dx)
    angle_right = math.atan2(dy_right, dx)

    visible = abs(angle_right - angle_left)
    return visible


def is_left_foot(body_part: str) -> bool:
    """Gibt True zurück, wenn Körperteil 'Left Foot' ist."""
    return body_part == "Left Foot"


def is_header(body_part: str) -> bool:
    """Gibt True zurück, wenn Körperteil 'Head' ist."""
    return body_part == "Head"
