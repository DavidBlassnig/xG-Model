import math

import numpy as np
import pandas as pd

from helpers import (
    GOAL_X,
    GOAL_CENTER_Y,
    distance_to_goal,
    angle_to_goal,
    angle_to_goal_rad,
    is_header,
    is_left_foot,
    player_blocked_angle,
    union_of_blocked_angles,
    blocked_goal_fraction,
    goalkeeper_free_zone_coverage,
)

FEATURE_SETS: dict[str, bool] = {
    "geometry": True,
    "body_part": True,
    "situation": True,
    "preceding": True,
    "goalkeeper": True,
    "pressure": True,
    "shot_lane": True,
}


# ---------------------------------------------------------------------------
# Individual feature-group functions
# ---------------------------------------------------------------------------

def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distanz und Winkel zum Tor."""
    df["distance_to_goal"] = df.apply(lambda r: distance_to_goal(r["x"], r["y"]), axis=1)
    df["angle_to_goal"] = df.apply(lambda r: angle_to_goal(r["x"], r["y"]), axis=1)
    df["angle_to_goal_rad"] = df.apply(lambda r: angle_to_goal_rad(r["x"], r["y"]), axis=1)
    return df


def add_body_part_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binäre Flags: is_header, is_left_foot."""
    bp = df["shot_body_part"].fillna("")
    df["is_header"] = bp.apply(is_header).astype(int)
    df["is_left_foot"] = bp.apply(is_left_foot).astype(int)
    return df


def add_situation_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot-Encoding der Spielsituation."""
    sit = df["shot_type"].fillna("Open Play")
    df["situation_open_play"] = (sit == "Open Play").astype(int)
    df["situation_free_kick"] = (sit == "Free Kick").astype(int)
    df["situation_corner"] = (sit == "From Corner").astype(int)
    df["situation_other"] = (~sit.isin(["Open Play", "Free Kick", "From Corner"])).astype(int)
    return df


def add_preceding_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vorhergehende Aktion via shot_key_pass_id.

    Erwartet vom Datenlade-Schritt angefügte Spalten mit Eigenschaften
    des Key-Pass-Events:
      kp_cross, kp_cut_back, kp_through_ball, kp_high_pass

    Schüsse ohne Key-Pass (shot_key_pass_id is NaN) erhalten
    preceding_no_assist=1; alle anderen Flags sind 0.
    """
    if "shot_key_pass_id" in df.columns:
        has_kp = df["shot_key_pass_id"].notna()
    else:
        has_kp = pd.Series(False, index=df.index)

    def _flag(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)
        return (has_kp & df[col].fillna(False).astype(bool)).astype(int)

    df["preceding_cross"] = _flag("kp_cross")
    df["preceding_cutback"] = _flag("kp_cut_back")
    df["preceding_through_ball"] = _flag("kp_through_ball")
    df["preceding_high_pass"] = _flag("kp_high_pass")
    df["preceding_no_assist"] = (~has_kp).astype(int)
    return df


def _parse_freeze_frame(ff):
    """Parse freeze_frame (list/ndarray/Iterable/NaN) into a flat list."""
    if ff is None:
        return []
    if isinstance(ff, (list, tuple)):
        return list(ff)
    if isinstance(ff, np.ndarray):
        return list(ff)
    # NaN scalar (float)
    if isinstance(ff, float):
        return []
    # Fallback: anything iterable
    try:
        return list(ff)
    except TypeError:
        return []


def _get(p, key, default=None):
    """dict.get-style accessor that also handles Mapping-like / structured objects."""
    if p is None:
        return default
    if isinstance(p, dict):
        return p.get(key, default)
    # Mapping-like (e.g. pyarrow StructScalar / numpy structured records)
    if hasattr(p, "get"):
        try:
            return p.get(key, default)
        except Exception:
            pass
    if hasattr(p, "__getitem__"):
        try:
            return p[key]
        except (KeyError, IndexError, TypeError):
            return default
    return default


def _player_xy(p):
    """Extract (x, y) from a freeze_frame player entry. None if invalid."""
    loc = _get(p, "location")
    if loc is None:
        return None
    if isinstance(loc, (list, tuple, np.ndarray)):
        try:
            if len(loc) >= 2:
                return float(loc[0]), float(loc[1])
        except (TypeError, ValueError):
            return None
    return None


def _is_goalkeeper(p):
    pos = _get(p, "position") or {}
    return _get(pos, "name") == "Goalkeeper"


def _is_teammate(p, default=True):
    val = _get(p, "teammate", default)
    return bool(val) if val is not None else default


def add_goalkeeper_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Torhüterposition aus shot.freeze_frame.
    Fehlende Werte mit Median imputen.
    """
    gk_x = []
    gk_y = []
    gk_dist = []

    ff_col = df.get("shot_freeze_frame", pd.Series([None] * len(df), index=df.index))

    for _, row in df.iterrows():
        ff = ff_col.get(row.name)
        players = _parse_freeze_frame(ff)
        found = False
        for p in players:
            if not _is_goalkeeper(p):
                continue
            if _is_teammate(p):
                continue
            xy = _player_xy(p)
            if xy is None:
                continue
            px, py = xy
            gk_x.append(px)
            gk_y.append(py)
            gk_dist.append(math.sqrt((GOAL_X - px) ** 2 + (GOAL_CENTER_Y - py) ** 2))
            found = True
            break
        if not found:
            gk_x.append(np.nan)
            gk_y.append(np.nan)
            gk_dist.append(np.nan)

    df["goalkeeper_x"] = gk_x
    df["goalkeeper_y"] = gk_y
    df["goalkeeper_distance_to_goal_center"] = gk_dist

    # Median imputation
    for col in ["goalkeeper_x", "goalkeeper_y", "goalkeeper_distance_to_goal_center"]:
        df[col] = df[col].fillna(df[col].median())

    return df


def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Druck durch Gegenspieler: n_defenders_in_cone, min_defender_distance.
    Cone: 3 Yards Öffnung in Richtung Tor.
    """
    n_def_in_cone = []
    min_def_dist = []

    ff_col = df.get("shot_freeze_frame", pd.Series([None] * len(df), index=df.index))

    for _, row in df.iterrows():
        ff = ff_col.get(row.name)
        players = _parse_freeze_frame(ff)
        sx, sy = row["x"], row["y"]

        defenders_in_cone = 0
        closest = np.inf

        for p in players:
            if _is_teammate(p):
                continue
            if _is_goalkeeper(p):
                continue
            xy = _player_xy(p)
            if xy is None:
                continue
            dx, dy = xy
            dist = math.sqrt((dx - sx) ** 2 + (dy - sy) ** 2)
            if dist < closest:
                closest = dist

            # Check if defender is in the cone between shooter and goal
            # Cone: within 3 yards laterally of the line to goal center
            vec_goal_x = GOAL_X - sx
            vec_goal_y = GOAL_CENTER_Y - sy
            vec_def_x = dx - sx
            vec_def_y = dy - sy
            goal_dist = math.sqrt(vec_goal_x ** 2 + vec_goal_y ** 2)
            if goal_dist > 0:
                # Project defender onto goal direction
                proj = (vec_def_x * vec_goal_x + vec_def_y * vec_goal_y) / goal_dist
                if proj > 0:  # defender is in front of shooter
                    # Perpendicular distance from defender to shot line
                    perp = abs(vec_def_x * vec_goal_y - vec_def_y * vec_goal_x) / goal_dist
                    if perp <= 3.0:
                        defenders_in_cone += 1

        n_def_in_cone.append(defenders_in_cone)
        min_def_dist.append(closest if closest != np.inf else np.nan)

    df["n_defenders_in_cone"] = n_def_in_cone
    df["min_defender_distance"] = min_def_dist

    # Median imputation for missing values
    for col in ["n_defenders_in_cone", "min_defender_distance"]:
        df[col] = df[col].fillna(df[col].median())

    return df


def add_shot_lane_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bewertet die freie Schussbahn auf Basis der Spielerpositionen im
    shot.freeze_frame (360°-Daten).

    Neue Spalten:
    - pct_goal_blocked:      Anteil der Torfläche, der durch gegnerische
                             Feldspieler verdeckt ist (0.0–1.0).
    - pct_goal_free:         Anteil der Torfläche, der frei von Feldspielern
                             ist (= 1 - pct_goal_blocked).
    - gk_free_zone_coverage: Wie gut der Torwart die freie Zone abdeckt
                             (0.0 = schlecht für TW, 1.0 = perfekt für TW).
    - net_open_goal_pct:     Freie Zone, die auch der Torwart nicht abdeckt
                             (= pct_goal_free * (1 - gk_free_zone_coverage)).

    Implementierungshinweise:
    - freeze_frame ist ein dict/list im StatsBomb-Format. Feldspieler filtern:
      freeze_frame-Einträge mit teammate=False und position != 'Goalkeeper'.
    - Torwart separat identifizieren: teammate=False, position == 'Goalkeeper'.
    - Fehlt freeze_frame oder ist er leer: alle vier Spalten mit np.nan füllen
      (nicht droppen — Imputation erfolgt downstream).
    - Koordinaten aus location-Feld jedes freeze_frame-Eintrags extrahieren.
    - Hilfsfunktionen aus helpers importieren:
      blocked_goal_fraction, player_blocked_angle,
      union_of_blocked_angles, goalkeeper_free_zone_coverage.
    """
    blocked_vals: list[float] = []
    free_vals: list[float] = []
    gk_cov_vals: list[float] = []
    net_open_vals: list[float] = []

    ff_col = df.get("shot_freeze_frame", pd.Series([None] * len(df), index=df.index))

    for _, row in df.iterrows():
        ff = ff_col.get(row.name)
        players = _parse_freeze_frame(ff)
        if not players:
            blocked_vals.append(np.nan)
            free_vals.append(np.nan)
            gk_cov_vals.append(np.nan)
            net_open_vals.append(np.nan)
            continue

        sx, sy = row["x"], row["y"]

        field_opponents: list[tuple[float, float]] = []
        gk_loc: tuple[float, float] | None = None
        for p in players:
            if _is_teammate(p):
                continue
            xy = _player_xy(p)
            if xy is None:
                continue
            if _is_goalkeeper(p):
                gk_loc = xy
            else:
                field_opponents.append(xy)

        blocked_frac = blocked_goal_fraction(sx, sy, field_opponents)
        free_frac = 1.0 - blocked_frac

        if gk_loc is not None:
            intervals = [
                player_blocked_angle(sx, sy, px, py)
                for (px, py) in field_opponents
            ]
            intervals = union_of_blocked_angles(intervals)
            gk_cov = goalkeeper_free_zone_coverage(sx, sy, gk_loc[0], gk_loc[1], intervals)
            net_open = free_frac * (1.0 - gk_cov)
        else:
            gk_cov = np.nan
            net_open = np.nan

        blocked_vals.append(blocked_frac)
        free_vals.append(free_frac)
        gk_cov_vals.append(gk_cov)
        net_open_vals.append(net_open)

    df["pct_goal_blocked"] = blocked_vals
    df["pct_goal_free"] = free_vals
    df["gk_free_zone_coverage"] = gk_cov_vals
    df["net_open_goal_pct"] = net_open_vals

    for col in ["pct_goal_blocked", "pct_goal_free", "gk_free_zone_coverage", "net_open_goal_pct"]:
        df[col] = df[col].fillna(df[col].median())

    return df

# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

_FEATURE_GROUP_FUNCS = {
    "geometry": add_geometry_features,
    "body_part": add_body_part_features,
    "situation": add_situation_features,
    "preceding": add_preceding_action_features,
    "goalkeeper": add_goalkeeper_features,
    "pressure": add_pressure_features,
    "shot_lane": add_shot_lane_features
}

_FEATURE_COLUMNS = {
    "geometry": ["distance_to_goal", "angle_to_goal", "angle_to_goal_rad"],
    "body_part": ["is_header", "is_left_foot"],
    "situation": ["situation_open_play", "situation_free_kick", "situation_corner", "situation_other"],
    "preceding": ["preceding_cross", "preceding_cutback", "preceding_through_ball", "preceding_high_pass", "preceding_no_assist"],
    "goalkeeper": ["goalkeeper_x", "goalkeeper_y", "goalkeeper_distance_to_goal_center"],
    "pressure": ["n_defenders_in_cone", "min_defender_distance"],
    "shot_lane": ["pct_goal_blocked", "pct_goal_free", "gk_free_zone_coverage", "net_open_goal_pct"],
}


def build_features(df: pd.DataFrame, feature_sets: dict[str, bool] | None = None) -> pd.DataFrame:
    """
    Hauptfunktion. Ruft alle aktiven Feature-Gruppen auf und gibt
    den vollständigen Feature-DataFrame zurück.
    """
    active = feature_sets if feature_sets is not None else FEATURE_SETS
    for name, enabled in active.items():
        if enabled and name in _FEATURE_GROUP_FUNCS:
            df = _FEATURE_GROUP_FUNCS[name](df)
    return df


def get_feature_columns(feature_sets: dict[str, bool] | None = None) -> list[str]:
    """Gibt die aktuelle, geordnete Liste aller Feature-Spaltennamen zurück."""
    active = feature_sets if feature_sets is not None else FEATURE_SETS
    cols = []
    for name, enabled in active.items():
        if enabled and name in _FEATURE_COLUMNS:
            cols.extend(_FEATURE_COLUMNS[name])
    return cols
