#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


#%%
# Cell 1: Setup and helpers
if HAS_SEABORN:
    sns.set_theme(style="whitegrid")
else:
    plt.style.use("ggplot")


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


#%%
# Cell 2: Load data
csv_path = Path("updated_turbidity_dataset.csv")
out_dir = Path("plots")
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

turbidity_cols = [
    "turbidity_raw",
    "turbidity_after_coagulation",
    "turbidity_after_sedimentation",
    "turbidity_after_filtration",
    "turbidity_final",
]


#%%
# Cell 3: Correlation heatmap (key parameters)
key_numeric = [
    "turbidity_raw",
    "coagulant_dose_mgL",
    "turbidity_after_coagulation",
    "turbidity_after_sedimentation",
    "turbidity_after_filtration",
    "turbidity_final",
    "rainfall_intensity_mm_hr",
    "TSS_mgL",
    "DOC_mgL",
    "flow_rate_m3s",
    "chemical_cost",
    "energy_consumption_kWh",
    "total_cost",
    "flocculation_time_min",
]
corr = df[key_numeric].corr(numeric_only=True)
plt.figure(figsize=(12, 9))
if HAS_SEABORN:
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
else:
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Heatmap (Key Parameters)")
savefig(out_dir / "01_correlation_heatmap.png")


#%%
# Cell 4: Top factors affecting final turbidity
final_corr = (
    df.select_dtypes(include=[np.number])
    .corr(numeric_only=True)["turbidity_final"]
    .drop("turbidity_final")
    .dropna()
)
top_final = final_corr.reindex(final_corr.abs().sort_values(ascending=False).index).head(12)
plt.figure(figsize=(10, 6))
colors = ["#d62728" if v < 0 else "#1f77b4" for v in top_final.values]
plt.barh(top_final.index[::-1], top_final.values[::-1], color=colors[::-1])
plt.axvline(0, color="black", linewidth=1)
plt.title("Top Factors Affecting Final Turbidity (Correlation)")
plt.xlabel("Correlation with turbidity_final")
savefig(out_dir / "02_top_factors_final_turbidity.png")
plt.show()


#%%
# Cell 5: Dependency scatter plots (strongest features vs final turbidity)
strongest_features = top_final.index[:6].tolist()
for i, feature in enumerate(strongest_features, start=1):
    plt.figure(figsize=(7, 5))
    x = df[feature].values
    y = df["turbidity_final"].values
    plt.scatter(x, y, s=8, alpha=0.3, color="#1f77b4")
    if np.isfinite(x).all() and np.isfinite(y).all():
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 200)
        y_line = m * x_line + b
        plt.plot(x_line, y_line, color="#d62728", linewidth=2)
    plt.title(f"{feature} vs turbidity_final")
    plt.xlabel(feature)
    plt.ylabel("turbidity_final")
    savefig(out_dir / f"03_dependency_{i}_{feature}.png")


#%%
# Cell 6: Turbidity progression through treatment stages (means)
stage_means = df[turbidity_cols].mean()
plt.figure(figsize=(9, 5))
plt.plot(stage_means.index, stage_means.values, marker="o", linewidth=2.5, color="#2ca02c")
for x, y in zip(stage_means.index, stage_means.values):
    plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
plt.title("Average Turbidity Across Treatment Stages")
plt.xlabel("Treatment stage")
plt.ylabel("Average turbidity")
plt.xticks(rotation=20)
savefig(out_dir / "04_turbidity_stagewise_means.png")


#%%
# Cell 7: Stage-wise turbidity reduction (absolute + percentage)
stage_pairs = [
    ("turbidity_raw", "turbidity_after_coagulation", "Raw -> Coagulation"),
    ("turbidity_after_coagulation", "turbidity_after_sedimentation", "Coagulation -> Sedimentation"),
    ("turbidity_after_sedimentation", "turbidity_after_filtration", "Sedimentation -> Filtration"),
    ("turbidity_after_filtration", "turbidity_final", "Filtration -> Final"),
    ("turbidity_raw", "turbidity_final", "Raw -> Final"),
]
labels = []
abs_drop = []
pct_drop = []
for src, dst, label in stage_pairs:
    src_mean = df[src].mean()
    dst_mean = df[dst].mean()
    drop = src_mean - dst_mean
    labels.append(label)
    abs_drop.append(drop)
    pct_drop.append((drop / src_mean) * 100 if src_mean != 0 else np.nan)

fig, ax1 = plt.subplots(figsize=(10, 6))
x_idx = np.arange(len(labels))
bars = ax1.bar(x_idx, abs_drop, color="#1f77b4", alpha=0.8, label="Absolute drop")
ax1.set_ylabel("Average turbidity drop (absolute)")
ax1.set_xticks(x_idx)
ax1.set_xticklabels(labels, rotation=20)
ax1.set_title("Stage-wise Turbidity Reduction (Absolute and %)")
for bar, v in zip(bars, abs_drop):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)

ax2 = ax1.twinx()
ax2.plot(x_idx, pct_drop, color="#d62728", marker="o", linewidth=2, label="% drop")
ax2.set_ylabel("Average turbidity drop (%)")
for x, v in zip(x_idx, pct_drop):
    ax2.text(x, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color="#d62728")
savefig(out_dir / "05_stagewise_reduction_absolute_percentage.png")


#%%
# Cell 8: Distribution comparison of turbidity stages
plt.figure(figsize=(10, 6))
if HAS_SEABORN:
    data = df[turbidity_cols].melt(var_name="stage", value_name="turbidity")
    sns.boxplot(data=data, x="stage", y="turbidity", showfliers=False)
else:
    plt.boxplot([df[c].values for c in turbidity_cols], labels=turbidity_cols, showfliers=False)
plt.title("Turbidity Distribution at Each Stage (Without Outliers)")
plt.xlabel("Stage")
plt.ylabel("Turbidity")
plt.xticks(rotation=20)
savefig(out_dir / "06_turbidity_stagewise_boxplot.png")


#%%
# Cell 9: Daily turbidity trends across stages
if df["timestamp"].notna().any():
    ts = df.set_index("timestamp")[turbidity_cols].resample("D").mean()
    plt.figure(figsize=(13, 6))
    for col in turbidity_cols:
        plt.plot(ts.index, ts[col], linewidth=1.3, label=col)
    plt.title("Daily Average Turbidity Through Treatment Stages")
    plt.xlabel("Date")
    plt.ylabel("Turbidity")
    plt.legend()
    savefig(out_dir / "07_daily_turbidity_trends.png")


#%%
# Cell 10: Final turbidity by scenario
plt.figure(figsize=(8, 5))
scenario_means = df.groupby("scenario_type", dropna=False)["turbidity_final"].mean().sort_values(ascending=False)
plt.bar(scenario_means.index, scenario_means.values, color="#9467bd")
for x, y in zip(scenario_means.index, scenario_means.values):
    plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
plt.title("Average Final Turbidity by Scenario Type")
plt.xlabel("Scenario")
plt.ylabel("Average turbidity_final")
savefig(out_dir / "08_final_turbidity_by_scenario.png")


#%%
# Cell 11: Final turbidity by season
plt.figure(figsize=(8, 5))
season_means = df.groupby("season", dropna=False)["turbidity_final"].mean().sort_values(ascending=False)
plt.bar(season_means.index, season_means.values, color="#ff7f0e")
for x, y in zip(season_means.index, season_means.values):
    plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
plt.title("Average Final Turbidity by Season")
plt.xlabel("Season")
plt.ylabel("Average turbidity_final")
savefig(out_dir / "09_final_turbidity_by_season.png")


#%%
# Cell 12: Pairwise dependency plot (optional, seaborn only)
selected = ["turbidity_final", "turbidity_raw", "TSS_mgL", "DOC_mgL", "coagulant_dose_mgL", "rainfall_intensity_mm_hr"]
if HAS_SEABORN:
    pair = sns.pairplot(df[selected], corner=True, plot_kws={"s": 8, "alpha": 0.25})
    pair.fig.suptitle("Pairwise Dependencies Around Final Turbidity", y=1.02)
    pair.savefig(out_dir / "10_pairplot_selected_features.png", dpi=180, bbox_inches="tight")
    plt.close("all")

print(f"Visualization complete. Plots saved in: {out_dir.resolve()}")


#%%
# Cell 13: Auto-generated insight report (presentation-ready text)
stage_values = df[turbidity_cols].mean()
raw_mean = stage_values["turbidity_raw"]
final_mean = stage_values["turbidity_final"]
overall_drop = raw_mean - final_mean
overall_drop_pct = (overall_drop / raw_mean) * 100 if raw_mean != 0 else np.nan

scenario_rank = df.groupby("scenario_type", dropna=False)["turbidity_final"].mean().sort_values(ascending=False)
season_rank = df.groupby("season", dropna=False)["turbidity_final"].mean().sort_values(ascending=False)

top_pos = top_final[top_final > 0].head(5)
top_neg = top_final[top_final < 0].head(3)

negative_final_count = int((df["turbidity_final"] < 0).sum())
missing_total = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

lines = []
lines.append("=== EDA INSIGHT REPORT ===")
lines.append("")
lines.append("1) Data quality")
lines.append(f"- Total rows: {len(df)}")
lines.append(f"- Total missing values: {missing_total}")
lines.append(f"- Duplicate rows: {duplicate_rows}")
lines.append(f"- Negative turbidity_final values: {negative_final_count}")
lines.append("")
lines.append("2) Final turbidity: strongest influencing factors (correlation)")
for feat, val in top_pos.items():
    lines.append(f"- Positive: {feat} (corr={val:.3f})")
for feat, val in top_neg.items():
    lines.append(f"- Negative: {feat} (corr={val:.3f})")
lines.append("")
lines.append("3) Stage-wise turbidity change (average)")
for src, dst, label in stage_pairs:
    src_mean = df[src].mean()
    dst_mean = df[dst].mean()
    drop = src_mean - dst_mean
    drop_pct = (drop / src_mean) * 100 if src_mean != 0 else np.nan
    lines.append(f"- {label}: {src_mean:.3f} -> {dst_mean:.3f} | drop={drop:.3f} ({drop_pct:.2f}%)")
lines.append(
    f"- Overall (Raw -> Final): {raw_mean:.3f} -> {final_mean:.3f} | "
    f"drop={overall_drop:.3f} ({overall_drop_pct:.2f}%)"
)
lines.append("")
lines.append("4) Scenario impact on final turbidity (higher is worse)")
for name, val in scenario_rank.items():
    lines.append(f"- {name}: {val:.3f}")
lines.append("")
lines.append("5) Seasonal impact on final turbidity (higher is worse)")
for name, val in season_rank.items():
    lines.append(f"- {name}: {val:.3f}")
lines.append("")
lines.append("6) Practical conclusion")
lines.append(
    "- Final turbidity is most dependent on downstream stage outputs, especially post-sedimentation and post-coagulation."
)
lines.append(
    "- Upstream load indicators (raw turbidity, TSS, DOC) also strongly affect final turbidity."
)
lines.append(
    "- Treatment train substantially reduces turbidity overall; scenario upsets and seasonal conditions still shift final outcomes."
)

report_text = "\n".join(lines)
print(report_text)

report_path = out_dir / "11_insight_report.txt"
report_path.write_text(report_text, encoding="utf-8")
print(f"\nInsight report saved to: {report_path.resolve()}")


#%%
# Cell 14: coagulant_dose_mgL vs turbidity_final (linear vs quadratic fit)
x = df["coagulant_dose_mgL"].values
y = df["turbidity_final"].values

mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

plt.figure(figsize=(9, 6))
plt.scatter(x, y, s=10, alpha=0.25, color="#1f77b4", label="Data")

# Linear fit
lin_coef = np.polyfit(x, y, 1)
lin_poly = np.poly1d(lin_coef)

# Quadratic fit
quad_coef = np.polyfit(x, y, 2)
quad_poly = np.poly1d(quad_coef)

x_line = np.linspace(np.min(x), np.max(x), 300)
y_lin = lin_poly(x_line)
y_quad = quad_poly(x_line)

plt.plot(x_line, y_lin, color="#2ca02c", linewidth=2, label="Linear fit")
plt.plot(x_line, y_quad, color="#d62728", linewidth=2.2, label="Quadratic fit")

# R^2 comparison
y_lin_pred = lin_poly(x)
y_quad_pred = quad_poly(x)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_lin = 1 - (np.sum((y - y_lin_pred) ** 2) / ss_tot) if ss_tot != 0 else np.nan
r2_quad = 1 - (np.sum((y - y_quad_pred) ** 2) / ss_tot) if ss_tot != 0 else np.nan

plt.title("coagulant_dose_mgL vs turbidity_final (Linear vs Quadratic)")
plt.xlabel("coagulant_dose_mgL")
plt.ylabel("turbidity_final")
plt.legend()
plt.text(
    0.02,
    0.98,
    f"R^2 linear = {r2_lin:.4f}\nR^2 quadratic = {r2_quad:.4f}",
    transform=plt.gca().transAxes,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
)
savefig(out_dir / "12_coagulant_vs_final_linear_quadratic.png")

print(
    "Saved: 12_coagulant_vs_final_linear_quadratic.png | "
    f"R^2 linear={r2_lin:.4f}, R^2 quadratic={r2_quad:.4f}"
)


#%%
# Cell 15: coagulant_dose_mgL vs turbidity_after_sedimentation (linear vs quadratic fit)
x2 = df["coagulant_dose_mgL"].values
y2 = df["turbidity_after_sedimentation"].values

mask2 = np.isfinite(x2) & np.isfinite(y2)
x2 = x2[mask2]
y2 = y2[mask2]

plt.figure(figsize=(9, 6))
plt.scatter(x2, y2, s=10, alpha=0.25, color="#1f77b4", label="Data")

# Linear fit
lin_coef2 = np.polyfit(x2, y2, 1)
lin_poly2 = np.poly1d(lin_coef2)

# Quadratic fit
quad_coef2 = np.polyfit(x2, y2, 2)
quad_poly2 = np.poly1d(quad_coef2)

x2_line = np.linspace(np.min(x2), np.max(x2), 300)
y2_lin = lin_poly2(x2_line)
y2_quad = quad_poly2(x2_line)

plt.plot(x2_line, y2_lin, color="#2ca02c", linewidth=2, label="Linear fit")
plt.plot(x2_line, y2_quad, color="#d62728", linewidth=2.2, label="Quadratic fit")

# R^2 comparison
y2_lin_pred = lin_poly2(x2)
y2_quad_pred = quad_poly2(x2)
ss_tot2 = np.sum((y2 - np.mean(y2)) ** 2)
r2_lin2 = 1 - (np.sum((y2 - y2_lin_pred) ** 2) / ss_tot2) if ss_tot2 != 0 else np.nan
r2_quad2 = 1 - (np.sum((y2 - y2_quad_pred) ** 2) / ss_tot2) if ss_tot2 != 0 else np.nan

plt.title("coagulant_dose_mgL vs turbidity_after_sedimentation (Linear vs Quadratic)")
plt.xlabel("coagulant_dose_mgL")
plt.ylabel("turbidity_after_sedimentation")
plt.legend()
plt.text(
    0.02,
    0.98,
    f"R^2 linear = {r2_lin2:.4f}\nR^2 quadratic = {r2_quad2:.4f}",
    transform=plt.gca().transAxes,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
)
savefig(out_dir / "13_coagulant_vs_sedimentation_linear_quadratic.png")

print(
    "Saved: 13_coagulant_vs_sedimentation_linear_quadratic.png | "
    f"R^2 linear={r2_lin2:.4f}, R^2 quadratic={r2_quad2:.4f}"
)


#%%
# Cell 16: coagulant_dose_mgL vs turbidity_after_coagulation (linear vs quadratic fit)
x3 = df["coagulant_dose_mgL"].values
y3 = df["turbidity_after_coagulation"].values

mask3 = np.isfinite(x3) & np.isfinite(y3)
x3 = x3[mask3]
y3 = y3[mask3]

plt.figure(figsize=(9, 6))
plt.scatter(x3, y3, s=10, alpha=0.25, color="#1f77b4", label="Data")

# Linear fit
lin_coef3 = np.polyfit(x3, y3, 1)
lin_poly3 = np.poly1d(lin_coef3)

# Quadratic fit
quad_coef3 = np.polyfit(x3, y3, 2)
quad_poly3 = np.poly1d(quad_coef3)

x3_line = np.linspace(np.min(x3), np.max(x3), 300)
y3_lin = lin_poly3(x3_line)
y3_quad = quad_poly3(x3_line)

plt.plot(x3_line, y3_lin, color="#2ca02c", linewidth=2, label="Linear fit")
plt.plot(x3_line, y3_quad, color="#d62728", linewidth=2.2, label="Quadratic fit")

# R^2 comparison
y3_lin_pred = lin_poly3(x3)
y3_quad_pred = quad_poly3(x3)
ss_tot3 = np.sum((y3 - np.mean(y3)) ** 2)
r2_lin3 = 1 - (np.sum((y3 - y3_lin_pred) ** 2) / ss_tot3) if ss_tot3 != 0 else np.nan
r2_quad3 = 1 - (np.sum((y3 - y3_quad_pred) ** 2) / ss_tot3) if ss_tot3 != 0 else np.nan

plt.title("coagulant_dose_mgL vs turbidity_after_coagulation (Linear vs Quadratic)")
plt.xlabel("coagulant_dose_mgL")
plt.ylabel("turbidity_after_coagulation")
plt.legend()
plt.text(
    0.02,
    0.98,
    f"R^2 linear = {r2_lin3:.4f}\nR^2 quadratic = {r2_quad3:.4f}",
    transform=plt.gca().transAxes,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
)
savefig(out_dir / "14_coagulant_vs_after_coagulation_linear_quadratic.png")

print(
    "Saved: 14_coagulant_vs_after_coagulation_linear_quadratic.png | "
    f"R^2 linear={r2_lin3:.4f}, R^2 quadratic={r2_quad3:.4f}"
)

# %%
