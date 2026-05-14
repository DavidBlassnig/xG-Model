# ⚽ xG Model — Expected Goals auf StatsBomb Open Data

Random Forest, XGBoost und XGBoost-PCA Expected-Goals-Modelle, trainiert auf
~88.000 Schüssen aus den StatsBomb Open Data. Vergleich gegen StatsBombs eigenes
xG-Modell. Interaktive Web-App zum Erkunden einzelner Matches und Schüsse.

---

## 🚀 Schnellstart

### 1. Repo klonen & Abhängigkeiten installieren

```bash
git clone <repo-url>
cd xG-model

python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# .\venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. StatsBomb Open Data lokal spiegeln (einmalig, ~1.5 GB)

```bash
git clone --depth=1 https://github.com/statsbomb/open-data.git data/statsbomb
```

`statsbombpy` liest die Daten dann direkt aus `data/statsbomb/data/` — keine API-Calls.

### 3. Modelle trainieren

Öffne `training.ipynb` in Jupyter oder VS Code und führe **alle Zellen der Reihe nach** aus.
Das Notebook:

1. Lädt ~88k Schüsse aus dem lokalen StatsBomb-Mirror (gecached als `data/shots_raw.parquet`)
2. Engineered 23 Features
3. Trainiert Random Forest, XGBoost und XGBoost-PCA
4. Speichert alle Artefakte unter `models/` für die Web-App

Laufzeit: ~10 Minuten für den Haupt-Pfad. Section 13 (5×10-Fold Cross-Validation
mit Wilcoxon-Test) braucht zusätzlich 30–60 Minuten und ist optional.

### 4. Web-App starten

```bash
streamlit run app.py
```

Öffne dann **<http://localhost:8501>** im Browser.

---

## 🖥 Web-App-Features

| | |
|---|---|
| **Sidebar** | Match-Auswahl: Dropdown mit allen Spielen ≥ 5 Schüssen, Format `#match_id — Team A vs Team B` |
| **Tab "📊 Match-Analyse"** | Zwei Pitches (eines pro Team) mit allen Schüssen (Größe ∝ xG, Farbe = Tor / kein Tor), Balkendiagramm mit xG-Summen aller 4 Modelle + tatsächliche Tore, sortierbare Schuss-Detail-Tabelle |
| **Tab "🎯 Einzelschuss"** | Pro Match einzelne Schuss-Auswahl. Zeigt Pitch mit Freeze-Frame-Spielern, alle 23 Feature-Werte, und die 4 xG-Vorhersagen als horizontale Bars |

---

## 📓 Notebook-Struktur (`training.ipynb`)

Alle Analysen sind in logisch aufeinander aufbauende Sections gegliedert.
Sie sollten **in dieser Reihenfolge** ausgeführt werden:

| Section | Inhalt | Zweck |
|--------:|--------|-------|
| **1**   | Setup & Imports | Environment vorbereiten |
| **2**   | Daten laden | Schüsse aus lokalem StatsBomb-Mirror (mit Progress-Bar), Cache in `data/shots_raw.parquet` |
| **3**   | Schüsse filtern & bereinigen | Penalties / Eigentore raus, Zielvariable `goal`, Koordinaten extrahieren |
| **4**   | Feature Engineering | Alle 23 Features über `features.py` bauen |
| **5**   | Train/Test-Split | Stratifiziert 80/20 nach Tor-Rate |
| **6**   | Modell: Random Forest | Baseline-Klassifikator |
| **7**   | Modell: XGBoost | Gradient Boosting, gleiche Features |
| **8**   | **Vergleich: RF + XGBoost vs. StatsBomb** | ROC-AUC, Brier, Log-Loss — die zentrale Baseline-Bewertung |
| **9**   | Kalibrierung & Residualanalyse | Calibration Curve mit 95 %-Wilson-Konfidenzbändern, Residualstreuung pro Modell |
| **10**  | Feature Importance (SHAP) | Globale Wichtigkeit aller Features im XGBoost |
| **11**  | PCA — Feature-Redundanz | Korrelations-Matrix, Scree-Plot, Loadings-Heatmap — zeigt wie viele PCs nötig sind |
| **12**  | PCA-XGBoost: 17 Komponenten | Pipeline `Standardize → PCA(17) → XGBoost`, Vergleich zu Section 8 |
| **13**  | **Ablation: Impact der Coverage-Features** | 5 Seeds × 10-Fold Cross-Validation + **Wilcoxon Signed-Rank Test** zwischen "PCA-17" und "ohne Coverage-Features" |
| **14**  | Deep-Dive: `net_open_goal_pct` | SHAP-Detail des wichtigsten Coverage-Features vs. Distanz / Winkel |
| **15**  | Modelle exportieren | Speichert RF, XGBoost, XGBoost-PCA + PCA-Pipeline unter `models/` (Voraussetzung für die Web-App) |

**Empfohlene Lese-Reihenfolge nach Interesse:**

- Wer nur den **Modellvergleich** sehen will → Sections 6–10
- Wer die **Coverage-Feature-Analyse** verstehen will → Sections 11–14
- Wer die **Web-App nutzen** will → komplettes Notebook bis inkl. Section 15

---

## 📁 Projektstruktur

```
xG-model/
├── README.md               ← dieses Dokument
├── CLAUDE.md               ← projekt-interne Konventionen
├── requirements.txt
├── app.py                  ← Streamlit Web-App
├── training.ipynb          ← Modell-Training & Analyse-Notebook
├── features.py             ← Feature Engineering (23 Features in 7 Gruppen)
├── helpers.py              ← Geometrie-Hilfsfunktionen (Distanz, Winkel, Coverage)
├── data/
│   ├── shots_raw.parquet   ← Cache aller Schüsse (von Notebook erzeugt)
│   └── statsbomb/          ← StatsBomb Open Data Mirror (eigener Git-Clone)
└── models/                 ← Vom Notebook erzeugt (Section 15)
    ├── random_forest.pkl
    ├── xgboost.json
    ├── xgboost_pca.json
    └── pca_pipeline.pkl
```

---

## 🛠 Häufige Stolpersteine

| Problem | Lösung |
|---|---|
| `STATSBOMB_DATA` Warnung beim Notebook-Start | Variable wird in Cell 1 gesetzt; achte darauf, dass `data/statsbomb/` existiert |
| Web-App startet, zeigt aber "Modell- oder Daten-Dateien fehlen" | Erst `training.ipynb` komplett laufen lassen, mindestens bis inkl. Section 15 |
| Section 13 dauert ewig | CV trainiert 100 XGBoost-Modelle (5 Seeds × 10 Folds × 2 Modelltypen). Du kannst sie überspringen, ohne dass die anderen Sections oder die Web-App leiden |
| Parquet liest freeze_frame als `np.ndarray` statt `list` | `features.py` behandelt beide Typen — bei NaN-Quote 100 % auf Coverage-Features: Kernel neu starten oder `importlib.reload(features)` |

---

## 📚 Quellen

- **StatsBomb Open Data**: <https://github.com/statsbomb/open-data>
- **statsbombpy** (Python-Wrapper): <https://github.com/statsbomb/statsbombpy>
- StatsBomb-xG: proprietäres Modell, dessen Predictions in den Open-Data-Events als `shot_statsbomb_xg` enthalten sind
