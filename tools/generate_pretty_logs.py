import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
OUT_DIR = RECORDS / "pretty_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe(row, key, default=""):
    v = row.get(key, default)
    return "" if v is None else str(v)


def block(title, row, extra_fields=None):
    extra_fields = extra_fields or []

    lines = []
    lines.append("=" * 90)
    lines.append(f"Experiment: {title}")
    lines.append(f"Run ID: {safe(row, 'run_id')}")
    lines.append(f"Dataset: {safe(row, 'dataset')}")
    lines.append(f"Missing rate: {safe(row, 'missing_rate')}")
    lines.append(f"Seed: {safe(row, 'seed')}")
    lines.append(f"Epochs: {safe(row, 'epochs')}")

    for label, key in extra_fields:
        lines.append(f"{label}: {safe(row, key)}")

    lines.append("-" * 90)
    lines.append("Results")
    lines.append(f"Train loss: {safe(row, 'train_loss')}")
    lines.append(f"Valid loss: {safe(row, 'valid_loss')}")
    lines.append(f"MAE: {safe(row, 'mae')}")
    lines.append(f"Corr: {safe(row, 'corr')}")
    lines.append(f"Acc-7: {safe(row, 'acc7')}")
    lines.append(f"Acc-5: {safe(row, 'acc5')}")
    lines.append(f"Acc-2 non-zero: {safe(row, 'acc2_non_zero')}")
    lines.append(f"F1 non-zero: {safe(row, 'f1_non_zero')}")
    lines.append(f"Acc-2: {safe(row, 'acc2')}")
    lines.append(f"F1: {safe(row, 'f1')}")
    lines.append(f"Note: {safe(row, 'note')}")
    lines.append("=" * 90)
    lines.append("")
    return "\n".join(lines)


def write_log(filename, title, rows, extra_fields=None):
    path = OUT_DIR / filename

    lines = []
    lines.append("#" * 90)
    lines.append(f"# {title}")
    lines.append(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("#" * 90)
    lines.append("")

    if not rows:
        lines.append("No records yet.")
    else:
        for row in rows:
            lines.append(block(title, row, extra_fields))

    path.write_text("\n".join(lines), encoding="utf-8")
    print("written:", path.relative_to(ROOT))


def main():
    hme_rows = read_csv(RECORDS / "02_hme_baseline_results.csv")
    topk_rows = read_csv(RECORDS / "03_hme_topk_results.csv")
    dare_rows = read_csv(RECORDS / "04_dare_hme_results.csv")

    write_log(
        "01_hme_baseline_pretty.log",
        "HME Baseline",
        hme_rows,
    )

    write_log(
        "02_hme_topk_pretty.log",
        "HME-TopK",
        topk_rows,
        extra_fields=[
            ("Top-K", "top_k"),
            ("Top temperature", "temperature"),
        ],
    )

    write_log(
        "03_dare_hme_pretty.log",
        "DARE-HME",
        dare_rows,
        extra_fields=[
            ("Degradation mode", "degradation_mode"),
            ("Quality module", "quality_module"),
            ("Uncertainty module", "uncertainty_module"),
        ],
    )

    all_lines = []
    for file in [
        "01_hme_baseline_pretty.log",
        "02_hme_topk_pretty.log",
        "03_dare_hme_pretty.log",
    ]:
        all_lines.append((OUT_DIR / file).read_text(encoding="utf-8"))
        all_lines.append("\n\n")

    all_path = OUT_DIR / "00_all_runs_pretty.log"
    all_path.write_text("".join(all_lines), encoding="utf-8")
    print("written:", all_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
