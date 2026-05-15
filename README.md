# xG Model — Expected Goals on StatsBomb Open Data

Random Forest, XGBoost, and XGBoost-PCA models trained on ~88k shots from
StatsBomb Open Data. Benchmarked against StatsBomb's own xG. Interactive
Streamlit app for exploring matches and individual shots.

---

## Quickstart

### 1. Clone & install

```bash
git clone <repo-url>
cd xG-model

python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# .\venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Mirror StatsBomb Open Data locally (one-time, ~1.5 GB)

```bash
git clone --depth=1 https://github.com/statsbomb/open-data.git data/statsbomb
```

`statsbombpy` reads directly from `data/statsbomb/data/` — no API calls.

### 3. Train models

Open `training.ipynb` and run all cells in order. The notebook loads ~88k
shots (cached to `data/shots_raw.parquet`), engineers 23 features, trains
all three models, and writes artifacts to `models/` for the app.

Runtime: ~10 minutes for the main path. Section 13.1 (20-seed
calibration sweep) adds ~15–25 minutes; Section 13.2 (5×10-fold CV +
Wilcoxon test) adds another 30–60 minutes. Both are optional.

note: You'll have to load the data either way. I decided to save locally because I wanted to do more stuff with it later.

### 4. Launch the app

```bash
streamlit run app.py
```

Open **<http://localhost:8501>**.

---

## App features

| | |
|---|---|
| **Sidebar** | Match picker — all games with ≥5 shots, formatted `#match_id — Team A vs Team B` |
| **Match Analysis tab** | Two pitches (one per team) showing all shots (size ∝ xG, color = goal/no goal), bar chart comparing xG totals across the 4 models vs. actual goals, sortable shot table |
| **Single Shot tab** | Per-match shot selector. Shows pitch with freeze-frame players, all 23 feature values, and the 4 xG predictions as horizontal bars |

---

## Notebook structure (`training.ipynb`)

Run sections in order:

| Section | Content |
|--------:|---------|
| **1**   | Setup & imports |
| **2**   | Load shots from local mirror (cached to `data/shots_raw.parquet`) |
| **3**   | Filter & clean (drop penalties/own goals, extract coordinates, build `goal` target) |
| **4**   | Feature engineering — 23 features via `features.py` |
| **5**   | Stratified 80/20 train/test split |
| **6**   | Random Forest baseline |
| **7**   | XGBoost |
| **8**   | **RF + XGBoost vs. StatsBomb** — ROC-AUC, Brier, log-loss |
| **9**   | Calibration & residual analysis (Wilson 95% CI bands) |
| **10**  | SHAP feature importance |
| **11**  | PCA — correlation matrix, scree plot, loadings heatmap |
| **12**  | PCA-XGBoost pipeline (18 components) vs. Section 8 |
| **13.1**| Calibration curves averaged across 20 random train/test splits (mean ± 1σ band per model) |
| **13.2**| **Coverage-features ablation** — 5 seeds × 10-fold CV + Wilcoxon signed-rank test |
| **14**  | Deep-dive on `net_open_goal_pct` — SHAP vs. distance/angle |
| **15**  | Export models to `models/` (required for the app) |

**Reading paths:**

- Model comparison only → Sections 6–10
- Coverage-feature analysis → Sections 11–14
- Web app → run through Section 15

---

## Project structure

```
xG-model/
├── README.md
├── CLAUDE.md               ← project conventions
├── requirements.txt
├── app.py                  ← Streamlit app
├── training.ipynb          ← training & analysis notebook
├── features.py             ← 23 features in 7 groups
├── helpers.py              ← geometry (distance, angle, coverage)
├── data/
│   ├── shots_raw.parquet   ← shot cache (notebook output)
│   └── statsbomb/          ← StatsBomb Open Data mirror
└── models/                 ← notebook output (Section 15)
    ├── random_forest.pkl
    ├── xgboost.json
    ├── xgboost_pca.json
    └── pca_pipeline.pkl
```

---

## 🛠 Troubleshooting

| Issue | Fix |
|---|---|
| `STATSBOMB_DATA` warning on notebook start | Variable is set in Cell 1 — make sure `data/statsbomb/` exists |
| App shows "model or data files missing" | Run `training.ipynb` through Section 15 first |
| Section 13 takes forever | Section 13.1 trains 60 XGBoost models, Section 13.2 trains 100. Both are safe to skip |
| Parquet reads `freeze_frame` as `np.ndarray` instead of `list` | `features.py` handles both — if Coverage features show 100% NaN, restart kernel or `importlib.reload(features)` |

---

## 📚 Sources

- **StatsBomb Open Data**: <https://github.com/statsbomb/open-data>
- **statsbombpy**: <https://github.com/statsbomb/statsbombpy>
- StatsBomb's xG predictions are in the events as `shot_statsbomb_xg`
