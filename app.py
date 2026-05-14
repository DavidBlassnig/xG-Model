"""
xG Model Explorer — Streamlit Web App.

Start mit:
    streamlit run app.py

Voraussetzung: training.ipynb wurde komplett ausgeführt, sodass die
Modelle unter models/ liegen und der Shot-Cache data/shots_raw.parquet existiert.
"""
import os
import pickle
from pathlib import Path

# STATSBOMB_DATA muss VOR dem statsbombpy-Import gesetzt sein
os.environ["STATSBOMB_DATA"] = str(Path(__file__).parent / "data" / "statsbomb" / "data")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Circle, Polygon
import streamlit as st
import xgboost as xgb

from features import build_features, get_feature_columns

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
SHOTS_CACHE = Path("data/shots_raw.parquet")
MODEL_DIR = Path("models")

st.set_page_config(
    page_title="xG Model Explorer",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Resource Loading (gecached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Lädt alle trainierten Modelle + PCA-Pipeline."""
    with open(MODEL_DIR / "random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_DIR / "xgboost.json")
    xgb_pca = xgb.XGBClassifier()
    xgb_pca.load_model(MODEL_DIR / "xgboost_pca.json")
    with open(MODEL_DIR / "pca_pipeline.pkl", "rb") as f:
        pca_pipeline = pickle.load(f)
    return rf, xgb_model, xgb_pca, pca_pipeline


@st.cache_data
def load_data():
    """Lädt Schüsse aus dem Parquet-Cache und appliziert dieselbe
    Bereinigung + Feature-Engineering wie das Notebook."""
    shots_raw = pd.read_parquet(SHOTS_CACHE)

    shots = shots_raw[shots_raw["shot_type"] != "Penalty"].copy()
    shots = shots[shots["shot_outcome"] != "Own Goal"].copy()
    shots["goal"] = (shots["shot_outcome"] == "Goal").astype(int)

    def _coord(loc, i):
        if loc is None:
            return np.nan
        try:
            if len(loc) >= 2:
                return float(loc[i])
        except TypeError:
            return np.nan
        return np.nan

    shots["x"] = shots["location"].apply(lambda loc: _coord(loc, 0))
    shots["y"] = shots["location"].apply(lambda loc: _coord(loc, 1))
    shots = shots.dropna(subset=["x", "y"])
    shots = build_features(shots)
    return shots, shots_raw


# ---------------------------------------------------------------------------
# Pitch-Drawing (StatsBomb-Koordinaten)
# ---------------------------------------------------------------------------
def draw_pitch(ax, facecolor="#1f7a3d", linecolor="white"):
    ax.set_facecolor(facecolor)
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, edgecolor=linecolor, lw=2))
    ax.plot([60, 60], [0, 80], color=linecolor, lw=2)
    ax.add_patch(Circle((60, 40), 10, fill=False, edgecolor=linecolor, lw=2))
    ax.add_patch(Circle((60, 40), 0.4, color=linecolor))
    for x0 in [0, 102]:
        ax.add_patch(Rectangle((x0, 18), 18, 44, fill=False, edgecolor=linecolor, lw=2))
    for x0 in [0, 114]:
        ax.add_patch(Rectangle((x0, 30), 6, 20, fill=False, edgecolor=linecolor, lw=2))
    for x_goal, off in [(0, -1.5), (120, 0)]:
        ax.add_patch(Rectangle((x_goal + off, 36), 1.5, 8, color=linecolor, alpha=0.7))
    for xs in [12, 108]:
        ax.add_patch(Circle((xs, 40), 0.4, color=linecolor))
    ax.add_patch(Arc((12, 40), 20, 20, angle=0, theta1=-53, theta2=53, color=linecolor, lw=2))
    ax.add_patch(Arc((108, 40), 20, 20, angle=0, theta1=127, theta2=233, color=linecolor, lw=2))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------------
# Match-Analyse Plot
# ---------------------------------------------------------------------------
def render_match_analysis(match_shots, team_a, team_b, goals_a, goals_b):
    fig = plt.figure(figsize=(16, 11), facecolor="#f7f7f7")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.32, wspace=0.18)

    for ax_idx, team, goals in [(0, team_a, goals_a), (1, team_b, goals_b)]:
        ax = fig.add_subplot(gs[0, ax_idx])
        draw_pitch(ax)
        team_shots = match_shots[match_shots["team"] == team]
        for _, s in team_shots.iterrows():
            is_goal = int(s["goal"]) == 1
            ax.scatter(s["x"], s["y"],
                       s=120 + 900 * max(s["avg_xg"], 0.02),
                       c="#2ca02c" if is_goal else "#d62728",
                       edgecolors="black", linewidths=1.2, alpha=0.78, zorder=5)
        ax.set_xlim(58, 123); ax.set_ylim(-2, 82)
        ax.set_title(f"{team}   ·   {goals} Tor(e)   ·   {len(team_shots)} Schuss/Schüsse",
                     fontsize=13, fontweight="bold", pad=10)

    ax_bar = fig.add_subplot(gs[1, :])
    model_names = ["Random Forest", "XGBoost", "XGBoost-PCA", "StatsBomb"]
    model_keys = ["rf_xg", "xgb_xg", "xgb_pca_xg", "sb_xg"]
    totals_a = {n: match_shots.loc[match_shots["team"] == team_a, k].sum()
                for n, k in zip(model_names, model_keys)}
    totals_b = {n: match_shots.loc[match_shots["team"] == team_b, k].sum()
                for n, k in zip(model_names, model_keys)}
    x_pos = np.arange(len(model_names))
    width = 0.38
    bars_a = ax_bar.bar(x_pos - width/2, [totals_a[m] for m in model_names],
                        width, color="#1f77b4", label=team_a, edgecolor="black")
    bars_b = ax_bar.bar(x_pos + width/2, [totals_b[m] for m in model_names],
                        width, color="#ff7f0e", label=team_b, edgecolor="black")
    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.04, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_bar.axhline(goals_a, color="#1f77b4", ls="--", lw=2, alpha=0.7)
    ax_bar.axhline(goals_b, color="#ff7f0e", ls="--", lw=2, alpha=0.7)
    y_max = max(max(totals_a.values()), max(totals_b.values()), goals_a, goals_b) * 1.18
    ax_bar.set_ylim(0, max(y_max, 1.5))
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(model_names, fontsize=11)
    ax_bar.set_ylabel("Total xG", fontsize=11)
    ax_bar.set_title(f"xG-Summe pro Team und Modell · Endstand: "
                     f"{team_a} {goals_a} – {goals_b} {team_b}",
                     fontsize=13, fontweight="bold", pad=10)
    ax_bar.legend(loc="upper left", fontsize=10)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_axisbelow(True)
    for s in ["top", "right"]:
        ax_bar.spines[s].set_visible(False)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Schuss-Analyse Plot
# ---------------------------------------------------------------------------
def render_shot_analysis(shot, feat_vals, predictions, feature_cols):
    rf_pred, xgb_pred, xgb_pca_pred, sb_pred = predictions
    actual = int(shot["goal"])

    fig = plt.figure(figsize=(18, 12), facecolor="#f7f7f7")
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.7], width_ratios=[2, 1, 1],
                          hspace=0.18, wspace=0.30)
    ax_pitch = fig.add_subplot(gs[0, :])
    ax_feat  = fig.add_subplot(gs[1, :2])
    ax_xg    = fig.add_subplot(gs[1, 2])

    draw_pitch(ax_pitch)
    sx, sy = float(shot["x"]), float(shot["y"])

    ax_pitch.add_patch(Polygon([(sx, sy), (120, 36), (120, 44)],
                               facecolor="#ffd700", alpha=0.20,
                               edgecolor="#ffd700", lw=1.5, linestyle="--", zorder=2))

    ff = shot.get("shot_freeze_frame")
    gk_pos = None
    tm_x, tm_y, op_x, op_y = [], [], [], []
    if isinstance(ff, (list, np.ndarray)):
        for p in ff:
            if not isinstance(p, dict):
                continue
            loc = p.get("location")
            if not isinstance(loc, (list, np.ndarray)) or len(loc) != 2:
                continue
            px, py = float(loc[0]), float(loc[1])
            is_tm = bool(p.get("teammate", False))
            is_gk = (p.get("position") or {}).get("name") == "Goalkeeper"
            if is_gk and not is_tm:
                gk_pos = (px, py)
            elif is_tm:
                tm_x.append(px); tm_y.append(py)
            else:
                op_x.append(px); op_y.append(py)

    if tm_x:
        ax_pitch.scatter(tm_x, tm_y, s=240, c="#1f77b4", edgecolors="white",
                         linewidths=2, zorder=5, label="Teammate")
    if op_x:
        ax_pitch.scatter(op_x, op_y, s=240, c="#d62728", edgecolors="white",
                         linewidths=2, zorder=5, label="Opponent")
    if gk_pos is not None:
        ax_pitch.scatter(*gk_pos, s=360, c="#ff8c00", marker="s", edgecolors="white",
                         linewidths=2.5, zorder=6, label="Goalkeeper")

    ax_pitch.scatter(sx, sy, s=650, c="#ffd700", edgecolors="black",
                     linewidths=2.5, zorder=10)
    ax_pitch.scatter(sx, sy, s=220, c="black", marker="*", zorder=11, label="Schütze")

    ax_pitch.set_xlim(58, 123); ax_pitch.set_ylim(-2, 82)

    player = shot.get("player", "?")
    team   = shot.get("team", "?")
    outcome = "GOAL ⚽" if actual == 1 else "NO GOAL"
    oc_color = "#2ca02c" if actual == 1 else "#d62728"

    ax_pitch.set_title(f"{player}   ·   {team}", fontsize=15, fontweight="bold", pad=10)
    ax_pitch.legend(loc="upper left", frameon=True, fontsize=10,
                    facecolor="white", framealpha=0.92)
    ax_pitch.text(0.99, 0.04, outcome, transform=ax_pitch.transAxes,
                  fontsize=18, fontweight="bold", color="white",
                  ha="right", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=oc_color, edgecolor="none"))

    # Feature panel
    ax_feat.axis("off"); ax_feat.set_xlim(0, 1); ax_feat.set_ylim(0, 1)
    ax_feat.set_title("Feature Values", fontsize=13, fontweight="bold", loc="left", pad=10)
    groups = {
        "Geometry":   ["distance_to_goal", "angle_to_goal", "angle_to_goal_rad"],
        "Body Part":  ["is_header", "is_left_foot"],
        "Situation":  ["situation_open_play", "situation_free_kick", "situation_corner", "situation_other"],
        "Preceding":  ["preceding_cross", "preceding_cutback", "preceding_through_ball", "preceding_high_pass", "preceding_no_assist"],
        "Goalkeeper": ["goalkeeper_x", "goalkeeper_y", "goalkeeper_distance_to_goal_center"],
        "Pressure":   ["n_defenders_in_cone", "min_defender_distance"],
        "Shot Lane":  ["pct_goal_blocked", "pct_goal_free", "gk_free_zone_coverage", "net_open_goal_pct"],
    }
    col_layout = [
        ["Geometry", "Body Part"],
        ["Situation", "Preceding"],
        ["Goalkeeper", "Pressure"],
        ["Shot Lane"],
    ]
    col_x_start = [0.00, 0.27, 0.54, 0.78]
    col_w = 0.22
    for col_i, group_names in enumerate(col_layout):
        y = 0.97
        for gname in group_names:
            ax_feat.text(col_x_start[col_i], y, gname,
                         fontsize=10.5, fontweight="bold", color="#222")
            y -= 0.08
            for f in groups[gname]:
                if f not in feat_vals.index:
                    continue
                v = feat_vals[f]
                if isinstance(v, (float, np.floating)) and not float(v).is_integer():
                    v_str = f"{v:.3f}"
                else:
                    v_str = f"{int(v)}"
                ax_feat.text(col_x_start[col_i] + 0.005, y, f,
                             fontsize=8.5, family="monospace", color="#555")
                ax_feat.text(col_x_start[col_i] + col_w, y, v_str,
                             fontsize=9.5, family="monospace",
                             color="#0066cc", fontweight="bold", ha="right")
                y -= 0.075
            y -= 0.025

    # xG bar chart
    ax_xg.set_title("xG-Vergleich", fontsize=13, fontweight="bold", loc="left", pad=10)
    names  = ["Random Forest", "XGBoost", "XGBoost-PCA", "StatsBomb"]
    vals   = [rf_pred, xgb_pred, xgb_pca_pred, sb_pred]
    colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]
    y_pos  = np.arange(len(names))
    bars = ax_xg.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=1, height=0.65)
    ax_xg.set_yticks(y_pos); ax_xg.set_yticklabels(names, fontsize=10.5)
    ax_xg.invert_yaxis()
    ax_xg.set_xlim(0, 1)
    ax_xg.set_xlabel("xG", fontsize=10)
    ax_xg.axvline(actual, color=oc_color, lw=2.2, ls="--", label=f"Outcome: {outcome}")
    for bar, v in zip(bars, vals):
        if v < 0.82:
            ax_xg.text(v + 0.02, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
                       va="center", ha="left", fontsize=10, fontweight="bold")
        else:
            ax_xg.text(v - 0.02, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
                       va="center", ha="right", fontsize=10, fontweight="bold", color="white")
    ax_xg.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax_xg.grid(axis="x", alpha=0.3); ax_xg.set_axisbelow(True)
    for s in ["top", "right"]:
        ax_xg.spines[s].set_visible(False)

    plt.tight_layout()
    return fig


# ===========================================================================
# UI
# ===========================================================================
st.title("⚽ xG Model Explorer")
st.markdown(
    "Interaktiver Vergleich von **Random Forest, XGBoost, XGBoost-PCA und StatsBomb-xG** "
    "auf Match- und Einzelschuss-Ebene."
)

# --- Resources laden ---
try:
    rf, xgb_model, xgb_pca, pca_pipeline = load_models()
    shots, shots_raw = load_data()
    feature_cols = get_feature_columns()
except FileNotFoundError as e:
    st.error(
        f"Modell- oder Daten-Dateien fehlen: `{e.filename}`\n\n"
        "Bitte zuerst `training.ipynb` von Anfang bis Ende ausführen, "
        "damit Cache und Modelle erzeugt werden."
    )
    st.stop()

# --- Sidebar: Match-Wahl ---
st.sidebar.header("🎯 Match wählen")

match_counts = shots.groupby("match_id").size().sort_values(ascending=False)
eligible_matches = match_counts[match_counts >= 5].index.tolist()


def match_label(mid):
    teams = shots_raw[shots_raw["match_id"] == mid]["team"].unique().tolist()
    return f"#{mid} — {' vs '.join(teams[:2])}"


match_id = st.sidebar.selectbox(
    "Match",
    options=eligible_matches,
    format_func=match_label,
    help="Spiele mit mindestens 5 Schüssen aus dem geladenen StatsBomb-Datensatz",
)
st.sidebar.caption(f"{len(eligible_matches)} Matches verfügbar")
st.sidebar.caption(f"Gesamt: {len(shots)} Schüsse")

# --- Aktuelle Match-Daten ---
match_shots = shots[shots["match_id"] == match_id].copy()
match_shots_raw = shots_raw[shots_raw["match_id"] == match_id].copy()
teams = list(match_shots_raw["team"].unique())
if len(teams) < 2:
    teams = teams + [t for t in match_shots["team"].unique() if t not in teams]
team_a, team_b = teams[0], teams[1]

def _goals(df, team):
    return int(((df["team"] == team) & (df["shot_outcome"] == "Goal")).sum())
goals_a = _goals(match_shots_raw, team_a)
goals_b = _goals(match_shots_raw, team_b)

# Predictions
X_match = match_shots[feature_cols]
match_shots = match_shots.assign(
    rf_xg      = rf.predict_proba(X_match)[:, 1],
    xgb_xg     = xgb_model.predict_proba(X_match)[:, 1],
    xgb_pca_xg = xgb_pca.predict_proba(pca_pipeline.transform(X_match))[:, 1],
    sb_xg      = match_shots["shot_statsbomb_xg"].fillna(0.0).values,
)
match_shots["avg_xg"] = match_shots[["rf_xg", "xgb_xg", "xgb_pca_xg", "sb_xg"]].mean(axis=1)

# --- Tab Layout ---
tab_match, tab_shot = st.tabs(["📊 Match-Analyse", "🎯 Einzelschuss"])

with tab_match:
    st.subheader(f"{team_a}  {goals_a} – {goals_b}  {team_b}")
    st.pyplot(render_match_analysis(match_shots, team_a, team_b, goals_a, goals_b))

    st.markdown("### Schuss-Details")
    detail_cols = ["minute", "team", "player", "shot_outcome",
                   "rf_xg", "xgb_xg", "xgb_pca_xg", "sb_xg"]
    existing = [c for c in detail_cols if c in match_shots.columns]
    detail = match_shots[existing].copy()
    for c in ["rf_xg", "xgb_xg", "xgb_pca_xg", "sb_xg"]:
        if c in detail.columns:
            detail[c] = detail[c].round(3)
    st.dataframe(detail.sort_values(["team", "minute"]).reset_index(drop=True),
                 use_container_width=True, height=400)

with tab_shot:
    st.markdown("Wähle einen Schuss aus diesem Match:")

    def shot_label(idx):
        s = match_shots.loc[idx]
        out = "⚽" if s["goal"] == 1 else "✗"
        return (f"{out}  Min {int(s['minute']):>3}  ·  "
                f"{s['team']:<25}  ·  {s.get('player', '?')}")

    shot_ids = match_shots.index.tolist()
    selected_idx = st.selectbox(
        "Schuss",
        options=shot_ids,
        format_func=shot_label,
        key=f"shot_select_{match_id}",
    )

    shot = match_shots.loc[selected_idx]
    feat_vals = match_shots.loc[selected_idx, feature_cols]

    X_one = match_shots.loc[[selected_idx], feature_cols]
    rf_pred  = float(rf.predict_proba(X_one)[0, 1])
    xgb_pred = float(xgb_model.predict_proba(X_one)[0, 1])
    xgb_pca_pred = float(xgb_pca.predict_proba(pca_pipeline.transform(X_one))[0, 1])
    sb_raw = shot.get("shot_statsbomb_xg")
    sb_pred = float(sb_raw) if not pd.isna(sb_raw) else 0.0

    st.pyplot(render_shot_analysis(
        shot, feat_vals,
        (rf_pred, xgb_pred, xgb_pca_pred, sb_pred),
        feature_cols,
    ))
