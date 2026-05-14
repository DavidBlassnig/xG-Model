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

PLAYER_WIDTH_YARDS = 0.6  # Angenommene Körperbreite eines Spielers in Yards (fix)

def player_blocked_angle(
    shooter_x: float, shooter_y: float,
    player_x: float, player_y: float,
    player_width: float = PLAYER_WIDTH_YARDS,
) -> tuple[float, float]:
    """
    Berechnet den durch einen Feldspieler blockierten Winkelbereich (in Grad)
    aus Sicht des Schützen auf das Tor.

    Gibt ein Tupel (angle_left, angle_right) zurück, das den blockierten
    Winkelkorridor auf der Torlinie beschreibt.
    Liegt der Spieler hinter dem Tor (player_x >= GOAL_X) oder hinter dem
    Schützen (player_x <= shooter_x), wird (0.0, 0.0) zurückgegeben.

    Berechnung:
    - Distanz Schütze → Spieler (euklidisch)
    - Halber Blockwinkel = arctan(player_width/2 / distanz)
    - Basiswinkel = arctan2(player_y - shooter_y, player_x - shooter_x)
    - Blockierter Bereich: basiswinkel ± halber Blockwinkel

    Implementation in Radians for consistency with angle_to_goal_rad.
    """
    if player_x >= GOAL_X or player_x <= shooter_x:
        return (0.0, 0.0)

    dx = player_x - shooter_x
    dy = player_y - shooter_y
    distance = math.sqrt(dx * dx + dy * dy)
    if distance <= 0.0:
        return (0.0, 0.0)

    base_angle = math.atan2(dy, dx)
    half_block = math.atan2(player_width / 2.0, distance)
    return (base_angle - half_block, base_angle + half_block)

def union_of_blocked_angles(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Vereinigt eine Liste überlappender Winkelintervalle zu einer
    disjunkten sortierten Liste. Wird verwendet, um Doppelzählung bei sich
    überschneidenden Spielerpositionen zu vermeiden.
    """
    cleaned = [(lo, hi) for (lo, hi) in intervals if hi > lo]
    if not cleaned:
        return []
    cleaned.sort(key=lambda iv: iv[0])
    merged: list[tuple[float, float]] = [cleaned[0]]
    for lo, hi in cleaned[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged

def blocked_goal_fraction(
    shooter_x: float, shooter_y: float,
    players: list[tuple[float, float]],
    player_width: float = PLAYER_WIDTH_YARDS,
) -> float:
    """
    Berechnet den Anteil der Torfläche (0.0–1.0), der durch Feldspieler
    (aus shot.freeze_frame) aus Sicht des Schützen blockiert ist.

    Vorgehensweise:
    1. Für jeden Feldspieler player_blocked_angle() aufrufen
    2. Alle Intervalle mit union_of_blocked_angles() vereinigen
    3. Gesamtwinkel des sichtbaren Tors (angle_to_goal_rad) als Basis nehmen
    4. Anteil = Summe blockierter Winkel / Gesamttorwinkel
       Ergebnis auf [0.0, 1.0] klippen.

    players: Liste von (x, y)-Koordinaten der gegnerischen Feldspieler
             (Torhüter separat behandeln — nicht in dieser Liste).
    """
    if shooter_x >= GOAL_X:
        return 0.0

    dx = GOAL_X - shooter_x
    goal_left = math.atan2(GOAL_POST_LEFT_Y - shooter_y, dx)
    goal_right = math.atan2(GOAL_POST_RIGHT_Y - shooter_y, dx)
    goal_lo, goal_hi = min(goal_left, goal_right), max(goal_left, goal_right)
    goal_span = goal_hi - goal_lo
    if goal_span <= 0.0:
        return 0.0

    clipped: list[tuple[float, float]] = []
    for (px, py) in players:
        lo, hi = player_blocked_angle(shooter_x, shooter_y, px, py, player_width)
        if hi <= lo:
            continue
        clo = max(lo, goal_lo)
        chi = min(hi, goal_hi)
        if chi > clo:
            clipped.append((clo, chi))

    union = union_of_blocked_angles(clipped)
    total_blocked = sum(hi - lo for (lo, hi) in union)
    fraction = total_blocked / goal_span
    return max(0.0, min(1.0, fraction))

def goalkeeper_free_zone_coverage(
    shooter_x: float, shooter_y: float,
    gk_x: float, gk_y: float,
    blocked_intervals: list[tuple[float, float]],
) -> float:
    """
    Bewertet, wie gut der Torwart die freie (= nicht durch Feldspieler
    blockierte) Torfläche abdeckt. Rückgabe: 0.0 (deckt freie Zone nicht ab)
    bis 1.0 (steht perfekt in der freien Zone).

    Vorgehensweise:
    1. Freie Winkelintervalle = Gesamttorwinkel minus blocked_intervals
    2. Mittelpunkt der freien Zone berechnen
    3. Winkel vom Schützen zum Torwart berechnen
    4. Nähe des Torwart-Winkels zum Mittelpunkt der freien Zone als Score
       (z. B. normierte Gaußfunktion oder lineare Interpolation über den
       freien Winkelbereich).
    """
    if shooter_x >= GOAL_X:
        return 0.0

    dx = GOAL_X - shooter_x
    goal_left = math.atan2(GOAL_POST_LEFT_Y - shooter_y, dx)
    goal_right = math.atan2(GOAL_POST_RIGHT_Y - shooter_y, dx)
    goal_lo, goal_hi = min(goal_left, goal_right), max(goal_left, goal_right)
    if goal_hi - goal_lo <= 0.0:
        return 0.0

    clipped: list[tuple[float, float]] = []
    for (lo, hi) in blocked_intervals:
        clo = max(lo, goal_lo)
        chi = min(hi, goal_hi)
        if chi > clo:
            clipped.append((clo, chi))
    clipped = union_of_blocked_angles(clipped)

    free: list[tuple[float, float]] = []
    cursor = goal_lo
    for (lo, hi) in clipped:
        if lo > cursor:
            free.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < goal_hi:
        free.append((cursor, goal_hi))

    if not free:
        return 0.0

    largest = max(free, key=lambda iv: iv[1] - iv[0])
    free_center = (largest[0] + largest[1]) / 2.0
    free_width = largest[1] - largest[0]

    gk_angle = math.atan2(gk_y - shooter_y, gk_x - shooter_x)

    sigma = max(free_width / 2.0, 1e-6)
    diff = gk_angle - free_center
    return math.exp(-(diff * diff) / (2.0 * sigma * sigma))