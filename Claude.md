# CLAUDE.md — Expected Goals (xG) Model

Dieses Dokument beschreibt Architektur, Konventionen und Workflows des xG-Projekts. Lies es vollständig, bevor du Code schreibst oder Dateien veränderst.

---

## Projektübersicht

Ziel ist die Entwicklung eines Expected-Goals-Modells auf Basis öffentlicher StatsBomb-Daten. Es werden zwei Modelle trainiert und verglichen:
- **Random Forest** (scikit-learn)
- **XGBoost**

Zusätzlich sollen professionelle xG-Referenzmodelle (z. B. StatsBomb xG, FBref xG) als Vergleichsbasis herangezogen werden.

---

## Umgebung

| Parameter | Wert |
|---|---|
| Editor | VS Code |
| Python | 3.14 |
| Paketmanager | pip (virtualenv empfohlen) |
| Notebook-Format | Jupyter (`training.ipynb`) |
| Datenquelle | StatsBomb Open Data (GitHub) |

### Abhängigkeiten (requirements.txt)

```
statsbombpy>=1.0.0
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
matplotlib>=3.8
seaborn>=0.13
shap>=0.44
jupyter>=1.0
ipywidgets>=8.0
```

---

## Projektstruktur

```
xg-model/
├── CLAUDE.md               ← dieses Dokument
├── requirements.txt
├── data/
│   └── statsbomb/          ← lokaler Spiegel des StatsBomb-Repos (s. u.)
│       ├── data/
│       │   ├── competitions.json
│       │   ├── matches/
│       │   ├── events/
│       │   ├── lineups/
│       │   └── three-sixty/
│       └── doc/
├── helpers.py              ← geometrische Hilfsfunktionen
├── features.py             ← Feature-Engineering
├── training.ipynb          ← Modelltraining & Evaluation
└── models/
    ├── random_forest.pkl
    └── xgboost.json
```

---

## Datenbeschaffung: StatsBomb Open Data

### Einmalig ausführen — Struktur vollständig erhalten

Das StatsBomb Open Data Repository (`https://github.com/statsbomb/open-data`) wird **einmalig geklont** und lokal unter `data/statsbomb/` gespeichert. Die Verzeichnisstruktur darf nicht verändert werden, da `statsbombpy` sie direkt erwartet.

```bash
# Im Projektverzeichnis ausführen (einmalig)
git clone --depth=1 https://github.com/statsbomb/open-data.git data/statsbomb
```

Danach muss `statsbombpy` so konfiguriert werden, dass es die lokalen Daten liest, **nicht** die API:

```python
# In jedem Skript/Notebook ganz oben einbinden
import os
os.environ["STATSBOMB_DATA"] = os.path.join(os.path.dirname(__file__), "data", "statsbomb", "data")
# Alternativ mit Path:
# from pathlib import Path
# os.environ["STATSBOMB_DATA"] = str(Path(__file__).parent / "data" / "statsbomb" / "data")
```

`statsbombpy` priorisiert `STATSBOMB_DATA` vor dem API-Fallback, wenn die Variable gesetzt ist.

### Daten aktualisieren (optional)

```bash
cd data/statsbomb && git pull
```

### Daten nie direkt commiten

Füge folgendes in `.gitignore` ein:

```
data/statsbomb/
models/*.pkl
models/*.json
```

---

## `helpers.py` — Geometrische Hilfsfunktionen

Diese Datei enthält **ausschließlich** reine, zustandslose Hilfsfunktionen ohne Side Effects. Keine Datenlade-Logik hier.

Alle Koordinaten folgen dem StatsBomb-Koordinatensystem:
- Spielfeld: 120 × 80 Yards
- Tor: Mitte bei x=120, y zwischen 36.0 und 44.0
- Tormittes: (120, 40)

### Pflichtfunktionen (implementieren)

```python
# helpers.py

GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
GOAL_POST_LEFT_Y = 36.0
GOAL_POST_RIGHT_Y = 44.0

def distance_to_goal(x: float, y: float) -> float:
    """Euklidische Distanz vom Schussort zur Tormitte."""
    ...

def angle_to_goal(x: float, y: float) -> float:
    """
    Schusswinkel in Grad (0° = keine Chance, 90° = direkt vor Tor).
    Berechnet den Winkel des sichtbaren Torpfostensegments.
    """
    ...

def angle_to_goal_rad(x: float, y: float) -> float:
    """Schusswinkel in Bogenmass."""
    ...

def is_left_foot(body_part: str) -> bool:
    """Gibt True zurück, wenn Körperteil 'Left Foot' ist."""
    ...

def is_header(body_part: str) -> bool:
    """Gibt True zurück, wenn Körperteil 'Head' ist."""
    ...
```

### Erweiterungshinweis

Neue Hilfsfunktionen kommen immer in `helpers.py`. `features.py` und `training.ipynb` importieren aus `helpers`, niemals umgekehrt.

---

## `features.py` — Feature-Engineering

Zentrale Stelle für alle Feature-Definitionen. Importiert aus `helpers.py`.

### Architekturprinzip: modulare Feature-Sets

Features sind in thematische Gruppen gegliedert. Jede Gruppe ist eine separate Funktion, die einen `pd.DataFrame` entgegennimmt und Spalten hinzufügt. So können neue Feature-Gruppen einfach ergänzt und einzeln aktiviert/deaktiviert werden.

```python
# features.py

from helpers import distance_to_goal, angle_to_goal, is_header

FEATURE_SETS: dict[str, bool] = {
    "geometry":       True,   # Distanz, Winkel
    "body_part":      True,   # Fuß / Kopf
    "situation":      True,   # Spielsituation
    "preceding":      True,   # Vorhergehende Aktion
    "goalkeeper":     True,   # Torhüterposition
    "pressure":       True,   # Gegnerdruck
    # NEU: hier schalten, bevor build_features() aufgerufen wird
}

def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distanz und Winkel zum Tor."""
    ...

def add_body_part_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binäre Flags: is_header, is_left_foot."""
    ...

def add_situation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot-Encoding der Spielsituation:
    open_play, free_kick, corner, penalty, other
    """
    ...

def add_preceding_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vorhergehende Aktion: cross (Flanke), dribble, counter, other.
    Aus dem StatsBomb-Feld shot.key_pass_id und events-Kontext ableiten.
    """
    ...

def add_goalkeeper_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Torhüterposition (wenn in shot.freeze_frame vorhanden):
    goalkeeper_x, goalkeeper_y, goalkeeper_distance_to_goal_center
    Fehlende Werte mit Median imputen — nicht droppen.
    """
    ...

def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Druck durch Gegenspieler:
    n_defenders_in_cone, min_defender_distance
    Cone-Radius: 3 Yards Öffnung in Richtung Tor.
    """
    ...

def build_features(df: pd.DataFrame, feature_sets: dict[str, bool] | None = None) -> pd.DataFrame:
    """
    Hauptfunktion. Ruft alle aktiven Feature-Gruppen auf und gibt
    den vollständigen Feature-DataFrame zurück.
    feature_sets=None → FEATURE_SETS aus Modul-Ebene verwenden.
    """
    ...

def get_feature_columns() -> list[str]:
    """Gibt die aktuelle, geordnete Liste aller Feature-Spaltennamen zurück."""
    ...
```

### Neue Features hinzufügen

1. Neue Funktion `add_<name>_features(df)` in `features.py` schreiben
2. Eintrag in `FEATURE_SETS` ergänzen (zunächst `False`, zum Testen `True`)
3. Aufruf in `build_features()` eintragen
4. `get_feature_columns()` aktualisieren
5. In `training.ipynb` Re-Training-Zelle ausführen → Vergleich mit vorherigem Modell

---

## `training.ipynb` — Modelltraining & Evaluation

Das Notebook ist in klar getrennte Sektionen unterteilt. Jede Sektion kann unabhängig ausgeführt werden.

### Notebook-Struktur (Sections)

```
1. Setup & Imports
2. Daten laden
3. Schüsse filtern & bereinigen
4. Feature Engineering
5. Train/Validation/Test-Split
6. Modell: Random Forest
7. Modell: XGBoost
8. Vergleich: professionelle xG-Modelle
9. Feature Importance (SHAP)
10. Residualanalyse & Kalibrierung
```

### 1. Setup & Imports

```python
import os
from pathlib import Path
os.environ["STATSBOMB_DATA"] = str(Path(".") / "data" / "statsbomb" / "data")

import statsbombpy.sb as sb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from features import build_features, get_feature_columns
```

### 2. Daten laden

```python
# Alle verfügbaren Competitions laden
competitions = sb.competitions()

# Schüsse aus allen Spielen laden (dauert, einmalig cachen)
# Tipp: competitions nach Wahl filtern, z. B. La Liga, Champions League
all_shots = []
for _, comp in competitions.iterrows():
    matches = sb.matches(competition_id=comp.competition_id, season_id=comp.season_id)
    for match_id in matches.match_id:
        events = sb.events(match_id=match_id)
        shots = events[events.type == "Shot"].copy()
        if not shots.empty:
            all_shots.append(shots)

shots_raw = pd.concat(all_shots, ignore_index=True)
```

### 3. Schüsse filtern & bereinigen

```python
# Penalties ausschließen (eigene xG-Berechnung: immer ~0.76)
shots = shots_raw[shots_raw["shot_type"] != "Penalty"].copy()

# Eigentore ausschließen
shots = shots[shots["shot_outcome"] != "Own Goal"].copy()

# Zielvariable
shots["goal"] = (shots["shot_outcome"] == "Goal").astype(int)

# Koordinaten aus Location extrahieren
shots["x"] = shots["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
shots["y"] = shots["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)

shots = shots.dropna(subset=["x", "y"])
```

### 5. Train/Validation/Test-Split

```python
# Zeitbasiert splitten, nicht zufällig — Data Leakage vermeiden
# (Saisons nach Zeit sortieren, letzten 20% als Testset)
# Falls keine Zeitinfo: stratified split nach goal-Rate

X = shots[get_feature_columns()].copy()
y = shots["goal"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 6. Modell: Random Forest

```python
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
```

### 7. Modell: XGBoost

```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    device="cpu",
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
```

### 8. Vergleich mit professionellen xG-Modellen

```python
# StatsBomb liefert sein xG direkt im shots-DataFrame:
sb_xg = shots.loc[X_test.index, "shot_statsbomb_xg"]

# Metriken-Vergleich
def evaluate_model(name, y_true, y_pred_proba):
    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_true, y_pred_proba),
        "Brier Score": brier_score_loss(y_true, y_pred_proba),
        "Log Loss": log_loss(y_true, y_pred_proba),
    }

results = pd.DataFrame([
    evaluate_model("Random Forest", y_test, rf_proba),
    evaluate_model("XGBoost", y_test, xgb_proba),
    evaluate_model("StatsBomb xG", y_test, sb_xg),
])
display(results)
```

### 9. Feature Importance mit SHAP

```python
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### 10. Kalibrierungsplot

```python
fig, ax = plt.subplots()
for name, proba in [("Random Forest", rf_proba), ("XGBoost", xgb_proba), ("StatsBomb xG", sb_xg)]:
    fraction_of_positives, mean_predicted = calibration_curve(y_test, proba, n_bins=10)
    ax.plot(mean_predicted, fraction_of_positives, marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--", label="Perfect")
ax.set_xlabel("Predicted xG")
ax.set_ylabel("Actual Goal Rate")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Datenqualität & Imputation

| Situation | Vorgehen |
|---|---|
| Fehlende Torhüterposition | Median über alle Schüsse aus ähnlicher Zone, **nicht** droppen |
| Fehlende Vorhergehende Aktion | Eigene Kategorie `"unknown"` |
| Ungültige Koordinaten (außerhalb Spielfeld) | Zeile droppen, loggen |
| Penalties | Aus Trainingsdaten ausschließen |

---

## Evaluation — Metriken

- **ROC-AUC**: Haupt-Diskriminierungsmetrik
- **Brier Score**: Kalibrierungsqualität (niedriger = besser)
- **Log Loss**: Trainingsverlust für XGBoost
- **Calibration Curve**: Visuell — zeigt ob Modell überschätzt oder unterschätzt

---

## Modelle speichern & laden

```python
import pickle, json

# Speichern
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)

xgb_model.save_model("models/xgboost.json")

# Laden
with open("models/random_forest.pkl", "rb") as f:
    rf_loaded = pickle.load(f)

xgb_loaded = xgb.XGBClassifier()
xgb_loaded.load_model("models/xgboost.json")
```

---

## Häufige Stolpersteine

| Problem | Ursache | Lösung |
|---|---|---|
| `STATSBOMB_DATA` ignoriert | Variable nach `import statsbombpy` gesetzt | Variable **vor** dem ersten Import setzen |
| Schüsse mit `x=None` | `location` ist kein List-Typ | Immer mit `isinstance(loc, list)` prüfen |
| `shot.freeze_frame` leer | Nicht alle Events haben 360°-Daten | Goalkeeper-Features mit `np.nan` befüllen, nicht droppen |
| Klassen-Ungleichgewicht | ~10% Tore, ~90% kein Tor | `class_weight="balanced"` (RF), `scale_pos_weight` (XGB) |
| Data Leakage | Zukünftige Spiele im Trainingsset | Split zeitbasiert oder nach Saison |

---

## Importkonventionen

```python
# helpers.py importiert: nur stdlib und numpy
# features.py importiert: helpers, pandas, numpy
# training.ipynb importiert: features, helpers, alle externen Pakete
```

`helpers.py` hat **keine** Abhängigkeit auf `features.py` oder externe Pakete außer numpy.

---

## Nächste Schritte / Roadmap

- [ ] `add_geometry_features` und `add_body_part_features` implementieren
- [ ] Datenladen aus StatsBomb testen (lokal)
- [ ] Baseline-Modelle trainieren (nur Distanz + Winkel)
- [ ] Alle Feature-Gruppen schrittweise aktivieren
- [ ] SHAP-Analyse für Feature-Selektion nutzen
- [ ] Professionellen xG-Vergleich in `training.ipynb` finalisieren
- [ ] Kalibrierung mit `CalibratedClassifierCV` verbessern
- [ ] Neues Feature: xG-Kette (Assist-Qualität)
- [ ] Neues Feature: Spielstand zum Zeitpunkt des Schusses