import csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"

OUT_SUMMARY_CSV = RECORDS / "07_pilot_mosi_3seed_summary.csv"
OUT_SUMMARY_MD = RECORDS / "07_pilot_mosi_3seed_summary.md"

OUT_DETAIL_CSV = RECORDS / "07_pilot_mosi_3seed_detail.csv"
OUT_DETAIL_MD = RECORDS / "07_pilot_mosi_3seed_detail.md"

OUT_IMPROVE_CSV = RECORDS / "07_pilot_mosi_3seed_improvement.csv"
OUT_IMPROVE_MD = RECORDS / "07_pilot_mosi_3seed_improvement.md"

OUT_PRETTY = RECORDS / "pretty_logs" / "04_pilot_mosi_3seed_summary.log"

TARGET_SEEDS = [5576, 1111, 2222]
TARGET_MR = [0.2, 0.3, 0.5]
TARGET_EPOCHS = 20


def read_csv(path, model_name):
    if not path.exists():
        print(f"missing file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["model_version"] = model_name
    return df


def fmt_mean_std(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def df_to_md(df, title, desc, path):
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(desc)
    lines.append("")

    if df.empty:
        lines.append("暂无数据。")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")

    for _, row in df.iterrows():
        vals = [str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(vals) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / "pretty_logs").mkdir(parents=True, exist_ok=True)

    hme = read_csv(RECORDS / "02_hme_baseline_results.csv", "HME")
    topk = read_csv(RECORDS / "03_hme_topk_results.csv", "HME-TopK")
    dare = read_csv(RECORDS / "04_dare_hme_results.csv", "DARE-HME")

    df = pd.concat([hme, topk, dare], ignore_index=True)

    if df.empty:
        print("No experiment records found.")
        return

    # 数值字段转换
    for col in ["missing_rate", "seed", "epochs", "mae", "corr", "acc2", "f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 只保留 MOSI 三 seed pilot 结果
    keep = (
        (df["dataset"].astype(str).str.upper() == "MOSI") &
        (df["run_id"].astype(str).str.startswith("pilot_mosi")) &
        (df["missing_rate"].isin(TARGET_MR)) &
        (df["seed"].isin(TARGET_SEEDS)) &
        (df["epochs"] == TARGET_EPOCHS)
    )
    df = df[keep].copy()

    # 去掉可能重复的 run_id
    df = df.drop_duplicates(subset=["run_id"], keep="last")

    # 排序
    model_order = {"HME": 0, "HME-TopK": 1, "DARE-HME": 2}
    df["model_order"] = df["model_version"].map(model_order)
    df = df.sort_values(["missing_rate", "model_order", "seed"])

    detail_cols = [
        "run_id", "dataset", "missing_rate", "model_version",
        "seed", "epochs", "mae", "corr", "acc2", "f1", "note"
    ]
    detail = df[detail_cols].copy()
    detail.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8")
    df_to_md(
        detail,
        "07 Pilot MOSI 3-Seed Detail",
        "MOSI 三个 seed 的 pilot 原始结果明细，只包含 mr=0.2 / 0.3 / 0.5。",
        OUT_DETAIL_MD
    )

    # 汇总 mean/std
    summary = (
        df.groupby(["missing_rate", "model_version"])
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

    # 保存数值版 CSV
    summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8")

    # 保存好看版 MD
    summary_md = summary.copy()
    summary_md["MAE ↓"] = summary_md.apply(lambda r: fmt_mean_std(r["mae_mean"], r["mae_std"]), axis=1)
    summary_md["Corr ↑"] = summary_md.apply(lambda r: fmt_mean_std(r["corr_mean"], r["corr_std"]), axis=1)
    summary_md["Acc-2 ↑"] = summary_md.apply(lambda r: fmt_mean_std(r["acc2_mean"], r["acc2_std"]), axis=1)
    summary_md["F1 ↑"] = summary_md.apply(lambda r: fmt_mean_std(r["f1_mean"], r["f1_std"]), axis=1)

    summary_md = summary_md[
        ["missing_rate", "model_version", "seed_count", "seeds", "MAE ↓", "Corr ↑", "Acc-2 ↑", "F1 ↑"]
    ]

    df_to_md(
        summary_md,
        "07 Pilot MOSI 3-Seed Summary",
        "结果格式为 mean±std。MAE 越低越好，Corr / Acc-2 / F1 越高越好。",
        OUT_SUMMARY_MD
    )

    # 计算 DARE 相对 HME / TopK 的提升
    rows = []
    for mr in TARGET_MR:
        part = summary[summary["missing_rate"] == mr].set_index("model_version")

        if "DARE-HME" not in part.index:
            continue

        dare_row = part.loc["DARE-HME"]

        for base in ["HME", "HME-TopK"]:
            if base not in part.index:
                continue

            base_row = part.loc[base]

            rows.append({
                "missing_rate": mr,
                "comparison": f"DARE-HME vs {base}",
                "mae_reduction": round(base_row["mae_mean"] - dare_row["mae_mean"], 4),
                "corr_gain": round(dare_row["corr_mean"] - base_row["corr_mean"], 4),
                "acc2_gain": round(dare_row["acc2_mean"] - base_row["acc2_mean"], 4),
                "acc2_gain_percent_point": round((dare_row["acc2_mean"] - base_row["acc2_mean"]) * 100, 2),
                "f1_gain": round(dare_row["f1_mean"] - base_row["f1_mean"], 4),
                "f1_gain_percent_point": round((dare_row["f1_mean"] - base_row["f1_mean"]) * 100, 2),
            })

    improve = pd.DataFrame(rows)
    improve.to_csv(OUT_IMPROVE_CSV, index=False, encoding="utf-8")
    df_to_md(
        improve,
        "07 Pilot MOSI 3-Seed Improvement",
        "这里统计 DARE-HME 相比 HME / HME-TopK 的平均提升。MAE reduction 越大越好，其余 gain 越大越好。",
        OUT_IMPROVE_MD
    )

    # 生成 pretty log
    pretty_lines = []
    pretty_lines.append("=" * 100)
    pretty_lines.append("Pilot MOSI 3-Seed Summary")
    pretty_lines.append("=" * 100)
    pretty_lines.append("")
    pretty_lines.append("Setting:")
    pretty_lines.append("- Dataset: MOSI")
    pretty_lines.append("- Seeds: 5576 / 1111 / 2222")
    pretty_lines.append("- Missing rates: 0.2 / 0.3 / 0.5")
    pretty_lines.append("- Epochs: 20")
    pretty_lines.append("- Models: HME / HME-TopK / DARE-HME")
    pretty_lines.append("")
    pretty_lines.append("Mean ± Std Results:")
    pretty_lines.append(summary_md.to_string(index=False))
    pretty_lines.append("")
    pretty_lines.append("DARE-HME Improvements:")
    pretty_lines.append(improve.to_string(index=False))
    pretty_lines.append("")
    pretty_lines.append("=" * 100)

    OUT_PRETTY.write_text("\n".join(pretty_lines), encoding="utf-8")

    print("Generated:")
    print(OUT_SUMMARY_CSV)
    print(OUT_SUMMARY_MD)
    print(OUT_DETAIL_CSV)
    print(OUT_DETAIL_MD)
    print(OUT_IMPROVE_CSV)
    print(OUT_IMPROVE_MD)
    print(OUT_PRETTY)

    print("\nPreview summary:")
    print(summary_md.to_string(index=False))


if __name__ == "__main__":
    main()
