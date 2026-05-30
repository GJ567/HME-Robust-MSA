import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
PRETTY = RECORDS / "pretty_logs"

TARGET_SEEDS = [5576, 1111, 2222]
TARGET_MR = [0.2, 0.3, 0.5]
TARGET_EPOCHS = 100

OUT_DETAIL_CSV = RECORDS / "08_official_mosi_detail.csv"
OUT_DETAIL_MD = RECORDS / "08_official_mosi_detail.md"

OUT_SUMMARY_CSV = RECORDS / "08_official_mosi_summary.csv"
OUT_SUMMARY_MD = RECORDS / "08_official_mosi_summary.md"

OUT_IMPROVE_CSV = RECORDS / "08_official_mosi_improvement.csv"
OUT_IMPROVE_MD = RECORDS / "08_official_mosi_improvement.md"

OUT_TOPK_CSV = RECORDS / "08_official_mosi_topk_mr05_retrieval_only.csv"
OUT_TOPK_MD = RECORDS / "08_official_mosi_topk_mr05_retrieval_only.md"

OUT_PRETTY_LOG = PRETTY / "05_official_mosi_summary.log"


def read_result(path, model_name):
    if not path.exists():
        print(f"missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["model_version"] = model_name

    # 兼容不同列名
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


def mean_std_text(mean, std):
    return f"{mean:.4f}±{std:.4f}"


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
    topk = read_result(RECORDS / "03_hme_topk_results.csv", "HME-TopK")
    dare = read_result(RECORDS / "04_dare_hme_results.csv", "DARE-HME")

    all_df = pd.concat([hme, topk, dare], ignore_index=True)

    if all_df.empty:
        print("No records found.")
        return

    all_df = to_number(
        all_df,
        ["missing_rate", "seed", "epochs", "mae", "corr", "acc2", "f1",
         "train_loss", "valid_loss", "acc7", "acc5"]
    )

    all_df["dataset"] = all_df["dataset"].astype(str).str.upper()
    all_df["run_id"] = all_df["run_id"].astype(str)

    # 只保留 official MOSI
    official = all_df[
        (all_df["dataset"] == "MOSI") &
        (all_df["run_id"].str.startswith("official_mosi")) &
        (all_df["missing_rate"].isin(TARGET_MR)) &
        (all_df["seed"].isin(TARGET_SEEDS)) &
        (all_df["epochs"] == TARGET_EPOCHS)
    ].copy()

    official = official.drop_duplicates(subset=["run_id"], keep="last")

    # 主实验只看 HME 和 DARE-HME
    main_df = official[official["model_version"].isin(["HME", "DARE-HME"])].copy()

    model_order = {"HME": 0, "HME-TopK": 1, "DARE-HME": 2}
    main_df["model_order"] = main_df["model_version"].map(model_order)
    main_df = main_df.sort_values(["missing_rate", "model_order", "seed"])

    detail_cols = [
        "run_id", "dataset", "missing_rate", "model_version",
        "seed", "epochs", "mae", "corr", "acc2", "f1", "note"
    ]
    detail = main_df[detail_cols].copy()
    detail.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8")
    save_md(
        detail,
        OUT_DETAIL_MD,
        "08 Official MOSI Detail",
        "MOSI official 主实验明细，只包含 HME 和 DARE-HME，mr=0.2 / 0.3 / 0.5，三组 seed。"
    )

    summary = (
        main_df
        .groupby(["missing_rate", "model_version"])
        .agg(
            seed_count=("seed", "nunique"),
            seeds=("seed", lambda x: "/".join(map(str, sorted(set(map(int, x)))))),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            corr_mean=("corr", "mean"),
            corr_std=("corr", "std"),
            acc2_mean=("acc2", "mean"),
            acc2_std=("acc2", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )

    summary["model_order"] = summary["model_version"].map(model_order)
    summary = summary.sort_values(["missing_rate", "model_order"]).drop(columns=["model_order"])
    summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8")

    summary_md = summary.copy()
    summary_md["MAE ↓"] = summary_md.apply(lambda r: mean_std_text(r["mae_mean"], r["mae_std"]), axis=1)
    summary_md["Corr ↑"] = summary_md.apply(lambda r: mean_std_text(r["corr_mean"], r["corr_std"]), axis=1)
    summary_md["Acc-2 ↑"] = summary_md.apply(lambda r: mean_std_text(r["acc2_mean"], r["acc2_std"]), axis=1)
    summary_md["F1 ↑"] = summary_md.apply(lambda r: mean_std_text(r["f1_mean"], r["f1_std"]), axis=1)

    summary_md = summary_md[
        ["missing_rate", "model_version", "seed_count", "seeds", "MAE ↓", "Corr ↑", "Acc-2 ↑", "F1 ↑"]
    ]

    save_md(
        summary_md,
        OUT_SUMMARY_MD,
        "08 Official MOSI Summary",
        "MOSI official 主实验汇总表。结果为 mean±std。MAE 越低越好，Corr / Acc-2 / F1 越高越好。"
    )

    # 提升表：DARE-HME vs HME
    improve_rows = []
    for mr in TARGET_MR:
        part = summary[summary["missing_rate"] == mr].set_index("model_version")
        if "HME" not in part.index or "DARE-HME" not in part.index:
            continue

        h = part.loc["HME"]
        d = part.loc["DARE-HME"]

        improve_rows.append({
            "missing_rate": mr,
            "comparison": "DARE-HME vs HME",
            "mae_reduction": round(h["mae_mean"] - d["mae_mean"], 4),
            "corr_gain": round(d["corr_mean"] - h["corr_mean"], 4),
            "acc2_gain": round(d["acc2_mean"] - h["acc2_mean"], 4),
            "acc2_gain_percent_point": round((d["acc2_mean"] - h["acc2_mean"]) * 100, 2),
            "f1_gain": round(d["f1_mean"] - h["f1_mean"], 4),
            "f1_gain_percent_point": round((d["f1_mean"] - h["f1_mean"]) * 100, 2),
        })

    improve = pd.DataFrame(improve_rows)
    improve.to_csv(OUT_IMPROVE_CSV, index=False, encoding="utf-8")
    save_md(
        improve,
        OUT_IMPROVE_MD,
        "08 Official MOSI Improvement",
        "DARE-HME 相比 HME 的平均提升。MAE reduction 越大越好，其余 gain 越大越好。"
    )

    # TopK 只整理 mr=0.5，作为 Retrieval-only 消融材料
    topk_df = official[
        (official["missing_rate"] == 0.5) &
        (official["model_version"].isin(["HME", "HME-TopK", "DARE-HME"]))
    ].copy()

    if not topk_df.empty:
        topk_df["model_order"] = topk_df["model_version"].map(model_order)
        topk_summary = (
            topk_df
            .groupby(["missing_rate", "model_version"])
            .agg(
                seed_count=("seed", "nunique"),
                seeds=("seed", lambda x: "/".join(map(str, sorted(set(map(int, x)))))),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                corr_mean=("corr", "mean"),
                corr_std=("corr", "std"),
                acc2_mean=("acc2", "mean"),
                acc2_std=("acc2", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
            )
            .reset_index()
        )
        topk_summary["model_order"] = topk_summary["model_version"].map(model_order)
        topk_summary = topk_summary.sort_values(["missing_rate", "model_order"]).drop(columns=["model_order"])

        topk_summary.to_csv(OUT_TOPK_CSV, index=False, encoding="utf-8")

        topk_md = topk_summary.copy()
        topk_md["MAE ↓"] = topk_md.apply(lambda r: mean_std_text(r["mae_mean"], r["mae_std"]), axis=1)
        topk_md["Corr ↑"] = topk_md.apply(lambda r: mean_std_text(r["corr_mean"], r["corr_std"]), axis=1)
        topk_md["Acc-2 ↑"] = topk_md.apply(lambda r: mean_std_text(r["acc2_mean"], r["acc2_std"]), axis=1)
        topk_md["F1 ↑"] = topk_md.apply(lambda r: mean_std_text(r["f1_mean"], r["f1_std"]), axis=1)

        topk_md = topk_md[
            ["missing_rate", "model_version", "seed_count", "seeds", "MAE ↓", "Corr ↑", "Acc-2 ↑", "F1 ↑"]
        ]

        save_md(
            topk_md,
            OUT_TOPK_MD,
            "08 Official MOSI Retrieval-only TopK Ablation",
            "这里把 HME-TopK 作为 Retrieval-only 中间版本，只用于说明单纯 TopK 检索增强的提升有限。"
        )
    else:
        topk_md = pd.DataFrame()

    # pretty log
    lines = []
    lines.append("=" * 100)
    lines.append("Official MOSI Summary")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Setting:")
    lines.append("- Dataset: MOSI")
    lines.append("- Missing rates: 0.2 / 0.3 / 0.5")
    lines.append("- Seeds: 5576 / 1111 / 2222")
    lines.append("- Epochs: 100")
    lines.append("- Main models: HME / DARE-HME")
    lines.append("- HME-TopK is kept only as Retrieval-only ablation at mr=0.5")
    lines.append("")
    lines.append("Main Results:")
    lines.append(summary_md.to_string(index=False))
    lines.append("")
    lines.append("Improvement:")
    lines.append(improve.to_string(index=False))
    lines.append("")

    if not topk_md.empty:
        lines.append("Retrieval-only / HME-TopK at mr=0.5:")
        lines.append(topk_md.to_string(index=False))
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
    print(OUT_TOPK_CSV)
    print(OUT_TOPK_MD)
    print(OUT_PRETTY_LOG)

    print("\nPreview:")
    print(summary_md.to_string(index=False))
    print("\nImprovement:")
    print(improve.to_string(index=False))


if __name__ == "__main__":
    main()
