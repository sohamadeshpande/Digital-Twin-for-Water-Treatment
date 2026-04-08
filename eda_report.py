import json
from pathlib import Path

import numpy as np
import pandas as pd


def series_json_safe(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    return value


def main():
    csv_path = Path("updated_turbidity_dataset.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape

    report = {
        "dataset_overview": {
            "file": str(csv_path),
            "rows": int(n_rows),
            "columns": int(n_cols),
            "column_names": list(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        }
    }

    dtype_summary = {c: str(t) for c, t in df.dtypes.items()}
    report["data_types"] = dtype_summary

    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100 if len(df) else missing
    report["missing_values"] = {
        c: {"missing_count": int(missing[c]), "missing_pct": round(float(missing_pct[c]), 3)}
        for c in df.columns
    }

    duplicate_rows = int(df.duplicated().sum())
    report["duplicates"] = {
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round((duplicate_rows / len(df)) * 100, 3) if len(df) else 0.0,
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    report["column_groups"] = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        for col in numeric_cols:
            row = desc.loc[col]
            numeric_summary[col] = {
                "count": series_json_safe(row["count"]),
                "mean": series_json_safe(row["mean"]),
                "std": series_json_safe(row["std"]),
                "min": series_json_safe(row["min"]),
                "p01": series_json_safe(row.get("1%")),
                "p05": series_json_safe(row.get("5%")),
                "p25": series_json_safe(row["25%"]),
                "p50_median": series_json_safe(row["50%"]),
                "p75": series_json_safe(row["75%"]),
                "p95": series_json_safe(row.get("95%")),
                "p99": series_json_safe(row.get("99%")),
                "max": series_json_safe(row["max"]),
                "skew": series_json_safe(df[col].skew()),
                "kurtosis": series_json_safe(df[col].kurt()),
            }
    report["numeric_summary"] = numeric_summary

    categorical_summary = {}
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        top_n = vc.head(10)
        categorical_summary[col] = {
            "unique_values": int(df[col].nunique(dropna=True)),
            "top_10_values": [
                {"value": series_json_safe(idx), "count": int(val), "pct": round((val / len(df)) * 100, 3) if len(df) else 0.0}
                for idx, val in top_n.items()
            ],
        }
    report["categorical_summary"] = categorical_summary

    outliers = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            outliers[col] = {"outlier_count_iqr": 0, "outlier_pct_iqr": 0.0}
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        count = int(mask.sum())
        outliers[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count_iqr": count,
            "outlier_pct_iqr": round((count / len(s)) * 100, 3),
        }
    report["outlier_analysis_iqr"] = outliers

    if numeric_cols:
        corr = df[numeric_cols].corr(numeric_only=True)
        corr_pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = corr.loc[c1, c2]
                if pd.notna(val):
                    corr_pairs.append((c1, c2, float(val), abs(float(val))))
        corr_pairs_sorted = sorted(corr_pairs, key=lambda x: x[3], reverse=True)
        report["top_correlations"] = [
            {"col_a": a, "col_b": b, "correlation": round(v, 4)}
            for a, b, v, _ in corr_pairs_sorted[:15]
        ]
    else:
        report["top_correlations"] = []

    target_candidates = [c for c in df.columns if "turbidity" in c.lower()]
    target_notes = {}
    for tgt in target_candidates:
        if tgt in numeric_cols:
            corr_to_target = (
                df[numeric_cols]
                .corr(numeric_only=True)[tgt]
                .drop(labels=[tgt], errors="ignore")
                .dropna()
                .sort_values(key=lambda x: np.abs(x), ascending=False)
            )
            target_notes[tgt] = [
                {"feature": idx, "corr_with_target": round(float(val), 4)}
                for idx, val in corr_to_target.head(10).items()
            ]
    report["target_candidate_analysis"] = target_notes

    json_path = Path("eda_report.json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"EDA report generated: {json_path.resolve()}")


if __name__ == "__main__":
    main()
