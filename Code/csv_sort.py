import pandas as pd

# -------- CONFIG --------
INPUT_CSV = "C:\\Users\\Michel Massad\\Desktop\\Robotics\\Code\\results_sift.csv"
OUT_SORTED = "results_sorted_sift.xlsx"
OUT_GOOD = "results_filtered_good_sift.xlsx"

# thresholds (tune if you want)
ROT_ERR_MAX = 10.0        # degrees
EULER_RMS_MAX = 6.0       # degrees
# ------------------------

# Load
df = pd.read_csv(INPUT_CSV)

# Ensure numeric
df["rot_err_deg"] = df["rot_err_deg"].astype(float)
df["euler_rms_err_deg"] = df["euler_rms_err_deg"].astype(float)

# 1) Sort by rotation error (worst first)
df_sorted = df.sort_values("rot_err_deg", ascending=False)

# Save sorted Excel
df_sorted.to_excel(OUT_SORTED, index=False)

# 2) Optional: filtered "good" results only
df_good = df[
    (df["rot_err_deg"] <= ROT_ERR_MAX) &
    (df["euler_rms_err_deg"] <= EULER_RMS_MAX)
].sort_values("rot_err_deg")

df_good.to_excel(OUT_GOOD, index=False)

# Console summary
print("=== EXPORT DONE ===")
print("Total rows:", len(df))
print("Sorted file:", OUT_SORTED)
print("Filtered good rows:", len(df_good))
print("Filtered file:", OUT_GOOD)

print("\nQuick stats (rot_err_deg):")
print("  median:", df["rot_err_deg"].median())
print("  p90:", df["rot_err_deg"].quantile(0.90))
print("  max:", df["rot_err_deg"].max())
