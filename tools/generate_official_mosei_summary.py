import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
PRETTY = RECORDS / "pretty_logs"

TARGET_MR = [0.2, 0.5]
TARGET_SEEDS = [5576]
TARGET_EPOCHS = 100

OUT_DETAIL_CSV = RECORDS / "09_official_mosei_detail.csv"
OUT_DETAIL_MD = RECORDS / "09_official_mosei_detail.md"

OUT_SUMMARY_CSV = RECORDS / "09_official_mosei_summary.csv"
OUT_SUMMARY_MD = RECORDS / "09_official_mosei_summary.md"

OUT_IMPROVE_CSV = RECORDS / "09_official_mosei_improvement.csv"
OUT_IMPROVE_MD = RECORDS / "09_official_mosei_improvement.md"

OUT_PRETTY_LOG = PRETTY / "06_official_mosei_summary.log"


def read_result(path, model_name):
    if not path.exists():
        print(f"missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["model_version"] = model_name

    if "f1" not in df.columns:
        if "f_score" in df.columns:
            df["f1"] = df["f_score"]
        elif "f_score_non_zero" in df.columns:
            df["f1"] = df["f_score_non_zero"]

    return df


def to_number(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_md(df, path, title, desc):
    lines = [f"# {title}", "", desc, ""]

    if df.empty:
        lines.append("暂无数据。")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")

    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    PRETTY.mkdir(parents=True, exist_ok=True)

    hme = read_result(RECORDS / "02_hme_baseline_results.csv", "HME")
    dare = read_result(RECORDS / "04_dare_hme_results.csv", "DARE-HME")

    df = pd.concat([hme, dare], ignore_index=True)

    if df.empty:
        print("No records found.")
        return

    df = to_number(
        df,
        ["missing_rate", "seed", "epochs", "mae", "corr", "acc2", "f1",
         "train_loss", "valid_loss", "acc7", "acc5"]
    )

    df["dataset"] = df["dataset"].astype(str).str.upper()
    df["run_id"] = df["run_id"].astype(str)

    keep = (
        (df["dataset"] == "MOSEI") &
        (df["run_id"].str.startswith("official_mosei")) &
        (df["missing_rate"].isin(TARGET_MR)) &
        (df["seed"].isin(TARGET_SEEDS)) &
        (df["epochs"] == TARGET_EPOCHS) &
        (df["model_version"].isin(["HME", "DARE-HME"]))
    )

    df = df[keep].copy()
    df = df.drop_duplicates(subset=["run_id"], keep="last")

    model_order = {"HME": 0, "DARE-HME": 1}
    df["model_order"] = df["model_version"].map(model_order)
    df = df.sort_values(["missing_rate", "model_order", "seed"])

    detail_cols = [
        "run_id", "dataset", "missing_rate", "model_version",
        "seed", "epochs", "mae", "corr", "acc2", "f1", "note"
    ]
    detail = df[detail_cols].copy()
    detail.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8")
    save_md(
        detail,
        OUT_DETAIL_MD,
        "09 Official MOSEI Detail",
        "MOSEI official 泛化实验明细。当前只包含 seed=5576，mr=0.2 / 0.5，HME 和 DARE-HME。"
    )

    summary = (
        df
        .groupby(["missing_rate", "model_version"])
        .agg(
            seed_count=("seed", "nunique"),
            seeds=("seed", lambda x: "/".join(map(str, sorted(set(map(int, x)))))),
            mae=("mae", "mean"),
            corr=("corr", "mean"),
            acc2=("acc2", "mean"),
            f1=("f1", "mean"),
        )
        .reset_index()
    )

    summary["model_order"] = summary["model_version"].map(model_order)
    summary = summary.sort_values(["missing_rate", "model_order"]).drop(columns=["model_order"])
    summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8")

    summary_md = summary.copy()
    for col in ["mae", "corr", "acc2", "f1"]:
        summary_md[col] = summary_md[col].map(lambda x: f"{x:.4f}")

    summary_md = summary_md.rename(columns={
        "missing_rate": "MR",
        "model_version": "Model",
        "seed_count": "Seed Count",
        "seeds": "Seeds",
        "mae": "MAE ↓",
        "corr": "Corr ↑",
        "acc2": "Acc-2 ↑",
        "f1": "F1 ↑",
    })

    save_md(
        summary_md,
        OUT_SUMMARY_MD,
        "09 Official MOSEI Summary",
        "MOSEI official 泛化验证表。MAE 越低越好，Corr / Acc-2 / F1 越高越好。"
    )

    # improvement
    rows = []
    for mr in TARGET_MR:
        part = summary[summary["missing_rate"] == mr].set_index("model_version")
        if "HME" not in part.index or "DARE-HME" not in part.index:
            continue

        h = part.loc["HME"]
        d = part.loc["DARE-HME"]

        rows.append({
            "MR": mr,
            "Comparison": "DARE-HME vs HME",
            "MAE reduction": round(h["mae"] - d["mae"], 4),
            "Corr gain": round(d["corr"] - h["corr"], 4),
            "Acc-2 gain": round(d["acc2"] - h["acc2"], 4),
            "Acc-2 gain percentage point": round((d["acc2"] - h["acc2"]) * 100, 2),
            "F1 gain": round(d["f1"] - h["f1"], 4),
            "F1 gain percentage point": round((d["f1"] - h["f1"]) * 100, 2),
        })

    improve = pd.DataFrame(rows)
    improve.to_csv(OUT_IMPROVE_CSV, index=False, encoding="utf-8")
    save_md(
        improve,
        OUT_IMPROVE_MD,
        "09 Official MOSEI Improvement",
        "DARE-HME 相比 HME 的提升。MAE reduction 越大越好，其余 gain 越大越好。"
    )

    lines = []
    lines.append("=" * 100)
    lines.append("Official MOSEI Summary")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Setting:")
    lines.append("- Dataset: MOSEI")
    lines.append("- Missing rates: 0.2 / 0.5")
    lines.append("- Seed: 5576")
    lines.append("- Epochs: 100")
    lines.append("- Models: HME / DARE-HME")
    lines.append("")
    lines.append("Main Results:")
    lines.append(summary_md.to_string(index=False))
    lines.append("")
    lines.append("Improvement:")
    lines.append(improve.to_string(index=False))
    lines.append("")
    lines.append("=" * 100)

    OUT_PRETTY_LOG.write_text("\n".join(lines), encoding="utf-8")

    print("Generated:")
    print(OUT_DETAIL_CSV)
    print(OUT_DETAIL_MD)
    print(OUT_SUMMARY_CSV)
    print(OUT_SUMMARY_MD)
    print(OUT_IMPROVE_CSV)
    print(OUT_IMPROVE_MD)
    print(OUT_PRETTY_LOG)

    print("\nPreview:")
    print(summary_md.to_string(index=False))
    print("\nImprovement:")
    print(improve.to_string(index=False))


if __name__ == "__main__":
    main()
