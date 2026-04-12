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
)

FEATURE_SETS: dict[str, bool] = {
    "geometry": True,
    "body_part": True,
    "situation": True,
    "preceding": True,
    "goalkeeper": True,
    "pressure": True,
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
    Vorhergehende Aktion abgeleitet aus shot_key_pass_id und events-Kontext.
    Nutzt shot_technique als Proxy für die Art des Angriffs.
    """
    tech = df.get("shot_technique", pd.Series("unknown", index=df.index)).fillna("unknown")
    df["preceding_cross"] = tech.str.contains("Volley", case=False, na=False).astype(int)
    df["preceding_head"] = tech.str.contains("Head", case=False, na=False).astype(int)
    df["preceding_lob"] = tech.str.contains("Lob", case=False, na=False).astype(int)
    df["preceding_normal"] = (~tech.isin(["Volley", "Head", "Lob"]) & (tech != "unknown")).astype(int)
    df["preceding_unknown"] = (tech == "unknown").astype(int)
    return df


def _parse_freeze_frame(ff):
    """Parse a freeze_frame value (list of dicts or NaN) into a list."""
    if isinstance(ff, list):
        return ff
    return []


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
            if p.get("position", {}).get("name") == "Goalkeeper" and not p.get("teammate", True):
                loc = p.get("location", [None, None])
                if isinstance(loc, list) and len(loc) == 2:
                    px, py = loc
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
            if p.get("teammate", True):
                continue
            if p.get("position", {}).get("name") == "Goalkeeper":
                continue
            loc = p.get("location", [None, None])
            if not isinstance(loc, list) or len(loc) != 2:
                continue
            dx, dy = loc[0], loc[1]
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
}

_FEATURE_COLUMNS = {
    "geometry": ["distance_to_goal", "angle_to_goal", "angle_to_goal_rad"],
    "body_part": ["is_header", "is_left_foot"],
    "situation": ["situation_open_play", "situation_free_kick", "situation_corner", "situation_other"],
    "preceding": ["preceding_cross", "preceding_head", "preceding_lob", "preceding_normal", "preceding_unknown"],
    "goalkeeper": ["goalkeeper_x", "goalkeeper_y", "goalkeeper_distance_to_goal_center"],
    "pressure": ["n_defenders_in_cone", "min_defender_distance"],
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
