import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"

NUM = r"(?:nan|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


INDEX_FIELDS = [
    "run_id", "date", "model_version", "dataset", "missing_rate", "seed",
    "epochs", "status", "result_table", "log_file", "note"
]

HME_FIELDS = [
    "run_id", "dataset", "missing_rate", "seed", "epochs",
    "train_loss", "valid_loss", "mae", "corr", "acc7", "acc5",
    "acc2_non_zero", "f1_non_zero", "acc2", "f1", "note"
]

TOPK_FIELDS = [
    "run_id", "dataset", "missing_rate", "top_k", "temperature", "seed", "epochs",
    "train_loss", "valid_loss", "mae", "corr", "acc7", "acc5",
    "acc2_non_zero", "f1_non_zero", "acc2", "f1", "note"
]

DARE_FIELDS = [
    "run_id", "dataset", "missing_rate", "degradation_mode",
    "quality_module", "uncertainty_module", "seed", "epochs",
    "train_loss", "valid_loss", "mae", "corr", "acc7", "acc5",
    "acc2_non_zero", "f1_non_zero", "acc2", "f1", "note"
]


def model_label(model: str) -> str:
    return {
        "hme": "HME",
        "topk": "HME-TopK",
        "dare": "DARE-HME",
    }[model]


def result_csv_for_model(model: str) -> Path:
    if model == "hme":
        return RECORDS / "02_hme_baseline_results.csv"
    if model == "topk":
        return RECORDS / "03_hme_topk_results.csv"
    if model == "dare":
        return RECORDS / "04_dare_hme_results.csv"
    raise ValueError(model)


def fields_for_model(model: str):
    if model == "hme":
        return HME_FIELDS
    if model == "topk":
        return TOPK_FIELDS
    if model == "dare":
        return DARE_FIELDS
    raise ValueError(model)


def read_rows(path: Path, fields):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    fixed = []
    for row in rows:
        fixed.append({k: row.get(k, "") for k in fields})
    return fixed


def write_rows(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def upsert_row(path: Path, fields, new_row):
    rows = read_rows(path, fields)
    updated = False
    for i, row in enumerate(rows):
        if row.get("run_id") == new_row.get("run_id"):
            rows[i] = {k: new_row.get(k, "") for k in fields}
            updated = True
            break
    if not updated:
        rows.append({k: new_row.get(k, "") for k in fields})
    write_rows(path, fields, rows)


def parse_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    # 去掉 tqdm / 终端进度条可能产生的 ANSI 控制符和回车刷新符
    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)
    text = text.replace("\r", "\n")
    flat = re.sub(r"\s+", " ", text)

    metrics = {
        "train_loss": "",
        "valid_loss": "",
        "mae": "",
        "corr": "",
        "acc7": "",
        "acc5": "",
        "acc2_non_zero": "",
        "f1_non_zero": "",
        "acc2": "",
        "f1": "",
    }

    epoch_re = re.compile(
        rf"epoch:\s*(\d+)\s*,\s*train_loss:\s*({NUM})\s*,\s*valid_loss:\s*({NUM})\s*,\s*test_acc2:\s*({NUM})",
        re.IGNORECASE,
    )
    epoch_matches = epoch_re.findall(flat)
    if epoch_matches:
        _, train_loss, valid_loss, test_acc2 = epoch_matches[-1]
        metrics["train_loss"] = train_loss
        metrics["valid_loss"] = valid_loss
        metrics["acc2"] = test_acc2

    best_valid_re = re.compile(rf"New best validation loss:\s*({NUM})", re.IGNORECASE)
    best_valid_matches = best_valid_re.findall(flat)
    if best_valid_matches:
        metrics["valid_loss"] = best_valid_matches[-1]

    metric_re = re.compile(
        rf"(best mae|current mae)\s*:\s*({NUM})\s*,\s*current corr\s*:\s*({NUM})\s*,\s*"
        rf"acc7\s*:\s*({NUM})\s*,\s*acc5\s*:\s*({NUM})\s*,\s*"
        rf"acc2_non_zero\s*:\s*({NUM})\s*,\s*f_score_non_zero\s*:\s*({NUM})\s*,\s*"
        rf"acc2\s*:\s*({NUM})\s*,\s*f_score\s*:\s*({NUM})",
        re.IGNORECASE,
    )
    metric_matches = metric_re.findall(flat)

    if metric_matches:
        best_matches = [m for m in metric_matches if m[0].lower() == "best mae"]
        m = best_matches[-1] if best_matches else metric_matches[-1]

        metrics["mae"] = m[1]
        metrics["corr"] = m[2]
        metrics["acc7"] = m[3]
        metrics["acc5"] = m[4]
        metrics["acc2_non_zero"] = m[5]
        metrics["f1_non_zero"] = m[6]
        metrics["acc2"] = m[7]
        metrics["f1"] = m[8]

    return metrics


def csv_to_md(csv_path: Path, md_path: Path, title: str, desc: str):
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    lines = [f"# {title}", "", desc, ""]

    if not fields:
        lines.append("暂无数据。")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")

    for row in rows:
        vals = []
        for field in fields:
            val = str(row.get(field, "")).replace("\n", " ").strip()
            vals.append(val)
        lines.append("| " + " | ".join(vals) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_all_md():
    csv_to_md(
        RECORDS / "01_run_index.csv",
        RECORDS / "01_run_index.md",
        "01 Run Index 总运行索引表",
        "这个表记录所有跑过的实验。具体指标看对应模型结果表。"
    )
    csv_to_md(
        RECORDS / "02_hme_baseline_results.csv",
        RECORDS / "02_hme_baseline_results.md",
        "02 HME Baseline Results",
        "这里只记录原始 HME baseline。"
    )
    csv_to_md(
        RECORDS / "03_hme_topk_results.csv",
        RECORDS / "03_hme_topk_results.md",
        "03 HME-TopK Results",
        "这里只记录 HME-TopK。"
    )
    csv_to_md(
        RECORDS / "04_dare_hme_results.csv",
        RECORDS / "04_dare_hme_results.md",
        "04 DARE-HME Results",
        "这里只记录 DARE-HME 完整模型。"
    )
    csv_to_md(
        RECORDS / "05_ablation_results.csv",
        RECORDS / "05_ablation_results.md",
        "05 Ablation Results",
        "这里记录消融实验。"
    )
    csv_to_md(
        RECORDS / "06_final_summary.csv",
        RECORDS / "06_final_summary.md",
        "06 Final Summary",
        "这里记录最终 mean ± std 汇总结果。"
    )


def ensure_empty_tables():
    RECORDS.mkdir(parents=True, exist_ok=True)

    if not (RECORDS / "01_run_index.csv").exists():
        write_rows(RECORDS / "01_run_index.csv", INDEX_FIELDS, [])

    if not (RECORDS / "02_hme_baseline_results.csv").exists():
        write_rows(RECORDS / "02_hme_baseline_results.csv", HME_FIELDS, [])

    if not (RECORDS / "03_hme_topk_results.csv").exists():
        write_rows(RECORDS / "03_hme_topk_results.csv", TOPK_FIELDS, [])

    if not (RECORDS / "04_dare_hme_results.csv").exists():
        write_rows(RECORDS / "04_dare_hme_results.csv", DARE_FIELDS, [])

    if not (RECORDS / "05_ablation_results.csv").exists():
        write_rows(
            RECORDS / "05_ablation_results.csv",
            [
                "run_id", "dataset", "missing_rate", "model_variant",
                "topk_enhance", "degradation_modeling", "quality_module",
                "uncertainty_module", "seed", "epochs", "mae", "corr",
                "acc2", "f1", "note"
            ],
            []
        )

    if not (RECORDS / "06_final_summary.csv").exists():
        write_rows(
            RECORDS / "06_final_summary.csv",
            [
                "dataset", "missing_rate", "model_version", "seed_count",
                "mae_mean", "mae_std", "corr_mean", "corr_std",
                "acc2_mean", "acc2_std", "f1_mean", "f1_std", "note"
            ],
            []
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_only", action="store_true")

    parser.add_argument("--run_id")
    parser.add_argument("--model", choices=["hme", "topk", "dare"])
    parser.add_argument("--dataset")
    parser.add_argument("--missing_rate")
    parser.add_argument("--seed")
    parser.add_argument("--epochs")
    parser.add_argument("--status", default="finished")
    parser.add_argument("--log_file")
    parser.add_argument("--note", default="")

    parser.add_argument("--top_k", default="")
    parser.add_argument("--temperature", default="")
    parser.add_argument("--degradation_mode", default="multi-level")
    parser.add_argument("--quality_module", default="on")
    parser.add_argument("--uncertainty_module", default="on")

    args = parser.parse_args()

    ensure_empty_tables()

    if args.refresh_only:
        refresh_all_md()
        print("Markdown tables refreshed.")
        return

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = ROOT / log_path

    metrics = parse_log(log_path) if log_path.exists() else {}

    model_version = model_label(args.model)
    result_csv = result_csv_for_model(args.model)
    result_fields = fields_for_model(args.model)

    result_table_name = result_csv.name
    log_rel = str(log_path.relative_to(ROOT)) if log_path.exists() and ROOT in log_path.parents else str(log_path)

    index_row = {
        "run_id": args.run_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model_version": model_version,
        "dataset": args.dataset.upper(),
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "epochs": args.epochs,
        "status": args.status,
        "result_table": result_table_name,
        "log_file": log_rel,
        "note": args.note,
    }
    upsert_row(RECORDS / "01_run_index.csv", INDEX_FIELDS, index_row)

    base_row = {
        "run_id": args.run_id,
        "dataset": args.dataset.upper(),
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_loss": metrics.get("train_loss", ""),
        "valid_loss": metrics.get("valid_loss", ""),
        "mae": metrics.get("mae", ""),
        "corr": metrics.get("corr", ""),
        "acc7": metrics.get("acc7", ""),
        "acc5": metrics.get("acc5", ""),
        "acc2_non_zero": metrics.get("acc2_non_zero", ""),
        "f1_non_zero": metrics.get("f1_non_zero", ""),
        "acc2": metrics.get("acc2", ""),
        "f1": metrics.get("f1", ""),
        "note": args.note,
    }

    if args.model == "topk":
        base_row["top_k"] = args.top_k
        base_row["temperature"] = args.temperature

    if args.model == "dare":
        base_row["degradation_mode"] = args.degradation_mode
        base_row["quality_module"] = args.quality_module
        base_row["uncertainty_module"] = args.uncertainty_module

    upsert_row(result_csv, result_fields, base_row)

    refresh_all_md()

    # Generate human-readable pretty logs
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_pretty_logs.py")],
        cwd=str(ROOT),
        check=False
    )

    print(f"Recorded result for run_id: {args.run_id}")
    print(f"Updated: {result_csv}")
    print("Markdown tables refreshed.")


if __name__ == "__main__":
    main()
