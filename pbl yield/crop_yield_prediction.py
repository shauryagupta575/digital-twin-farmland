"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CROP YIELD PREDICTION — ML PIPELINE                               ║
║           Architecture: Random Forest + Gradient Boosting + Ridge           ║
║           Target: Yield_kg_per_ha                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, json, os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH  = r'D:\pbl yield\Merged_Dataset.csv'
OUTPUT_DIR = r'D:\pbl yield\model_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    'Year', 'Area_ha',
    'N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha',
    'Temperature_C', 'Humidity_%', 'pH',
    'Rainfall_mm', 'Wind_Speed_m_s', 'Solar_Radiation_MJ_m2_day',
    'Crop_enc'
]
TARGET = 'Yield_kg_per_ha'

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD & EXPLORE
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1 — DATA LOADING & EXPLORATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\n📦 Dataset shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"🌾 Crops           : {sorted(df['Crop'].unique())}")
print(f"📅 Year range      : {df['Year'].min()} – {df['Year'].max()}")
print(f"🗺  States          : {df['State Name'].nunique()} states")
print(f"📍 Districts       : {df['Dist Name'].nunique()} districts")
print(f"\n📊 Target stats:")
print(df[TARGET].describe().round(2).to_string())
print(f"\n✅ Missing values  : {df.isnull().sum().sum()}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 2 — PREPROCESSING")
print("=" * 65)

le = LabelEncoder()
df['Crop_enc'] = le.fit_transform(df['Crop'])
print(f"\n🔠 Crop encoding   : {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📂 Train size      : {len(X_train):,}")
print(f"📂 Test  size      : {len(X_test):,}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────────────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 3 — MODEL TRAINING")
print("=" * 65)

models = {
    'Random Forest': {
        'model': RandomForestRegressor(n_estimators=200, max_depth=None,
                                       min_samples_split=2, random_state=42, n_jobs=-1),
        'scaled': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                           max_depth=5, random_state=42),
        'scaled': False
    },
    'Ridge Regression': {
        'model': Ridge(alpha=1.0),
        'scaled': True
    },
}

results = {}
for name, cfg in models.items():
    Xtr = X_train_sc if cfg['scaled'] else X_train
    Xte = X_test_sc  if cfg['scaled'] else X_test
    cfg['model'].fit(Xtr, y_train)
    preds = cfg['model'].predict(Xte)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    results[name] = {'model': cfg['model'], 'preds': preds,
                     'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"\n  ▶ {name}")
    print(f"      RMSE : {rmse:>10.2f} kg/ha")
    print(f"      MAE  : {mae:>10.2f} kg/ha")
    print(f"      R²   : {r2:>10.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. BEST MODEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['R2'])
best      = results[best_name]
print(f"\n🏆 Best model      : {best_name}")
print(f"   R²             : {best['R2']:.6f}")
print(f"   RMSE           : {best['RMSE']:.2f} kg/ha")

# ──────────────────────────────────────────────────────────────────────────────
# 5. CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 4 — CROSS-VALIDATION (5-Fold)")
print("=" * 65)
kf  = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best['model'], X, y, cv=kf, scoring='r2', n_jobs=-1)
print(f"\n  R² per fold      : {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean R²          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. SAVE MODEL ARTEFACTS
# ──────────────────────────────────────────────────────────────────────────────
joblib.dump(best['model'], f'{OUTPUT_DIR}/best_model.pkl')
joblib.dump(scaler,        f'{OUTPUT_DIR}/scaler.pkl')
joblib.dump(le,            f'{OUTPUT_DIR}/label_encoder.pkl')
meta = {
    'best_model': best_name, 'features': FEATURE_COLS,
    'crops': list(le.classes_),
    'R2': best['R2'], 'RMSE': best['RMSE'], 'MAE': best['MAE'],
    'cv_mean_r2': float(cv_scores.mean()),
    'cv_std_r2':  float(cv_scores.std())
}
with open(f'{OUTPUT_DIR}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\n💾 Model saved to  : {OUTPUT_DIR}/best_model.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 5 — GENERATING PLOTS")
print("=" * 65)

# Plot A — Actual vs Predicted (3 models)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Crop Yield Prediction — Actual vs Predicted', fontsize=15, fontweight='bold')
palette = ['#2ecc71', '#3498db', '#e74c3c']
for ax, (name, res), c in zip(axes, results.items(), palette):
    mn = min(y_test.min(), res['preds'].min())
    mx = max(y_test.max(), res['preds'].max())
    ax.scatter(y_test, res['preds'], alpha=0.25, s=8, color=c)
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Perfect fit')
    ax.set_title(f"{name}\nR²={res['R2']:.4f}  RMSE={res['RMSE']:.0f}", fontsize=11)
    ax.set_xlabel('Actual Yield (kg/ha)')
    ax.set_ylabel('Predicted Yield (kg/ha)')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ actual_vs_predicted.png")

# Plot B — Feature Importance
rf_model = results['Random Forest']['model']
imp = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
bar_colors = ['#e74c3c' if v == imp.max() else '#3498db' for v in imp.values]
imp.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
for i, v in enumerate(imp.values):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ feature_importance.png")

# Plot C — Avg Yield by Crop
fig, ax = plt.subplots(figsize=(9, 5))
crop_yield = df.groupby('Crop')[TARGET].mean().sort_values(ascending=False)
colors_c = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax.bar(crop_yield.index, crop_yield.values, color=colors_c, edgecolor='white', width=0.55)
ax.set_title('Average Yield by Crop Type', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Yield (kg/ha)')
for bar, val in zip(bars, crop_yield.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f'{val:,.0f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/yield_by_crop.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ yield_by_crop.png")

# Plot D — Model Metrics Comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics  = ['RMSE', 'MAE', 'R2']
mcolors  = ['#e74c3c', '#f39c12', '#2ecc71']
for ax, metric, mc in zip(axes, metrics, mcolors):
    vals  = [results[m][metric] for m in results]
    names = list(results.keys())
    bars  = ax.bar(names, vals, color=mc, alpha=0.85, edgecolor='white')
    ax.set_title(metric, fontweight='bold', fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
fig.suptitle('Model Comparison Metrics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ model_metrics.png")

# Plot E — Yield Distribution
fig, ax = plt.subplots(figsize=(10, 5))
for crop, color in zip(df['Crop'].unique(), ['#2ecc71','#3498db','#e74c3c','#f39c12']):
    subset = df[df['Crop'] == crop][TARGET]
    ax.hist(subset, bins=50, alpha=0.55, label=crop.capitalize(), color=color)
ax.set_title('Yield Distribution by Crop', fontsize=13, fontweight='bold')
ax.set_xlabel('Yield (kg/ha)')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/yield_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ yield_distribution.png")

# ──────────────────────────────────────────────────────────────────────────────
# 8. PREDICTION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def predict_yield(year, area_ha, N_req, P_req, K_req,
                  temperature, humidity, pH, rainfall,
                  wind_speed, solar_radiation, crop_name):
    """
    Predict crop yield for given input parameters.

    Parameters
    ----------
    year           : int   — Cultivation year
    area_ha        : float — Area in hectares
    N_req          : float — Nitrogen requirement (kg/ha)
    P_req          : float — Phosphorus requirement (kg/ha)
    K_req          : float — Potassium requirement (kg/ha)
    temperature    : float — Temperature in °C
    humidity       : float — Relative humidity (%)
    pH             : float — Soil pH
    rainfall       : float — Rainfall in mm
    wind_speed     : float — Wind speed (m/s)
    solar_radiation: float — Solar radiation (MJ/m²/day)
    crop_name      : str   — One of: rice, maize, chickpea, cotton

    Returns
    -------
    float — Predicted yield in kg/ha
    """
    model = joblib.load(f'{OUTPUT_DIR}/best_model.pkl')
    le_   = joblib.load(f'{OUTPUT_DIR}/label_encoder.pkl')
    crop_enc = le_.transform([crop_name.lower()])[0]
    row = np.array([[year, area_ha, N_req, P_req, K_req,
                     temperature, humidity, pH, rainfall,
                     wind_speed, solar_radiation, crop_enc]])
    return float(model.predict(row)[0])

0
# ── Demo prediction ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 6 — SAMPLE PREDICTION")
print("=" * 65)
sample_pred = predict_yield(
    year=2024, area_ha=10000, N_req=18.0, P_req=8.0, K_req=11.3,
    temperature=22, humidity=70, pH=6.0, rainfall=800,
    wind_speed=2.5, solar_radiation=20, crop_name='maize'
)
print(f"\n  🌽 Maize yield prediction (2024) : {sample_pred:,.2f} kg/ha")
print(f"\n{'=' * 65}")
print("  ✅ Pipeline complete! All files saved to: model_output/")
print("=" * 65)
