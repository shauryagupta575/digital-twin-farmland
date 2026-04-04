"""
╔══════════════════════════════════════════════════════════════╗
║   SMART DATASET MERGER — FIXED VERSION                       ║
╚══════════════════════════════════════════════════════════════╝
"""
import pandas as pd
import numpy as np
import os

# ── PATHS ───────────────────────────────────────────────────────
OLD_DATA    = r'D:\pbl yield\Custom_Crops_yield_Historical_Dataset.csv'
NEW_DATA    = r"D:\pbl yield\All-India_-Crop-wise-Area,-Production-&-Yield.csv"
MERGED_DATA = r'D:\pbl yield\Merged_Dataset.csv'

print("=" * 60)
print("  SMART DATASET MERGER")
print("=" * 60)

# ── Check files exist before loading ────────────────────────────
for path in [OLD_DATA, NEW_DATA]:
    if not os.path.exists(path):
        print(f"\n❌ File not found: {path}")
        print("   Please make sure this file is in D:\\pbl yield\\")
        exit()

# ── Load both files ─────────────────────────────────────────────
print("\n📂 Loading datasets...")
df_old = pd.read_csv(OLD_DATA)
df_new = pd.read_csv(NEW_DATA)
print(f"   Old dataset : {len(df_old):,} rows  |  {df_old['Crop'].nunique()} crops")
print(f"   New dataset : {len(df_new):,} rows  |  {df_new['Crop'].nunique()} crops")

# ── Keep only Total season rows ─────────────────────────────────
df_new = df_new[df_new['Season'] == 'Total'].copy()
print(f"   New dataset (Total season only) : {len(df_new)} rows")

# ── Crop name mapping ───────────────────────────────────────────
crop_map = {
    'Rice': 'rice', 'Maize': 'maize', 'Wheat': 'wheat',
    'Cotton': 'cotton', 'Soybean': 'soybean', 'Sugarcane': 'sugarcane',
    'Groundnut': 'groundnut', 'Bajra': 'bajra', 'Jowar': 'jowar',
    'Gram': 'chickpea', 'Barley': 'barley', 'Tur': 'tur',
    'Urad': 'urad', 'Moong': 'moong', 'Lentil': 'lentil',
    'Ragi': 'ragi', 'Jute': 'jute', 'Tobacco': 'tobacco',
}

# ── Get default soil/weather values per crop from old data ──────
print("\n⚙️  Computing default values per crop...")
crop_defaults = df_old.groupby('Crop').agg({
    'N_req_kg_per_ha': 'mean', 'P_req_kg_per_ha': 'mean',
    'K_req_kg_per_ha': 'mean', 'Temperature_C': 'mean',
    'Humidity_%': 'mean', 'pH': 'mean', 'Rainfall_mm': 'mean',
    'Wind_Speed_m_s': 'mean', 'Solar_Radiation_MJ_m2_day': 'mean'
}).round(2)

global_avg = df_old.agg({
    'N_req_kg_per_ha': 'mean', 'P_req_kg_per_ha': 'mean',
    'K_req_kg_per_ha': 'mean', 'Temperature_C': 'mean',
    'Humidity_%': 'mean', 'pH': 'mean', 'Rainfall_mm': 'mean',
    'Wind_Speed_m_s': 'mean', 'Solar_Radiation_MJ_m2_day': 'mean'
}).round(2)

# ── Convert new dataset rows into old format ────────────────────
print("\n🔄 Converting new dataset to matching format...")
rows = []
for _, row in df_new.iterrows():
    crop_raw  = row['Crop']
    crop_name = crop_map.get(crop_raw, crop_raw.lower().replace(' ', '_'))

    if crop_name in crop_defaults.index:
        defaults = crop_defaults.loc[crop_name]
    else:
        defaults = global_avg

    for year in [2021, 2022, 2023, 2024]:
        area_col  = f'Area-{year}-{str(year+1)[-2:]}'
        yield_col = f'Yield-{year}-{str(year+1)[-2:]}'

        area_val  = row.get(area_col, np.nan)
        yield_val = row.get(yield_col, np.nan)

        if pd.isna(area_val) or pd.isna(yield_val):
            continue

        area_ha = area_val * 100000

        rows.append({
            'Dist Code': 0,   'Year': year,
            'State Code': 0,  'State Name': 'All India',
            'Dist Name': 'All India', 'Crop': crop_name,
            'Area_ha': round(area_ha, 1),
            'Yield_kg_per_ha': round(yield_val, 2),
            'N_req_kg_per_ha': defaults['N_req_kg_per_ha'],
            'P_req_kg_per_ha': defaults['P_req_kg_per_ha'],
            'K_req_kg_per_ha': defaults['K_req_kg_per_ha'],
            'Total_N_kg': round(defaults['N_req_kg_per_ha'] * area_ha, 2),
            'Total_P_kg': round(defaults['P_req_kg_per_ha'] * area_ha, 2),
            'Total_K_kg': round(defaults['K_req_kg_per_ha'] * area_ha, 2),
            'Temperature_C': defaults['Temperature_C'],
            'Humidity_%': defaults['Humidity_%'],
            'pH': defaults['pH'],
            'Rainfall_mm': defaults['Rainfall_mm'],
            'Wind_Speed_m_s': defaults['Wind_Speed_m_s'],
            'Solar_Radiation_MJ_m2_day': defaults['Solar_Radiation_MJ_m2_day'],
        })

df_converted = pd.DataFrame(rows)
print(f"   Converted rows : {len(df_converted)}")

# ── Merge and save ──────────────────────────────────────────────
print("\n🔗 Merging datasets...")
df_merged = pd.concat([df_old, df_converted], ignore_index=True)
df_merged.drop_duplicates(inplace=True)
df_merged.to_csv(MERGED_DATA, index=False)

print("\n" + "=" * 60)
print("  ✅ MERGE COMPLETE!")
print("=" * 60)
print(f"\n  Old rows        : {len(df_old):,}")
print(f"  New rows added  : {len(df_converted):,}")
print(f"  Total rows      : {len(df_merged):,}")
print(f"  Year range      : {df_merged['Year'].min()} – {df_merged['Year'].max()}")
print(f"  Total crops     : {df_merged['Crop'].nunique()}")
print(f"\n  Crops now:")
for crop in sorted(df_merged['Crop'].unique()):
    print(f"    • {crop}")
print(f"\n  Saved to : {MERGED_DATA}")
print(f"\n  ✅ Next step:")
print(f"     Open crop_yield_prediction.py")
print(f"     Change DATA_PATH to: D:\\pbl yield\\Merged_Dataset.csv")
print(f"     Then run it to retrain!")