import os
import re
import sys
import csv
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "ablation"
RECORDS = ROOT / "records"
PRETTY = RECORDS / "pretty_logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
RECORDS.mkdir(parents=True, exist_ok=True)
PRETTY.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
MAIN = ROOT / "HME_MSA" / "HME_main_dare.py"

OUT_CSV = RECORDS / "10_ablation_mosi_mr05_s5576_results.csv"
OUT_MD = RECORDS / "10_ablation_mosi_mr05_s5576_results.md"
OUT_SUMMARY_MD = RECORDS / "10_ablation_mosi_mr05_s5576_summary.md"
OUT_PRETTY = PRETTY / "07_ablation_mosi_mr05_s5576.log"

COMMON_ARGS = [
    str(MAIN),
    "--dataset", "mosi",
    "--learning_rate", "2e-5",
    "--d_l", "64",
    "--missing_rate", "0.5",
    "--layers", "1",
    "--hyper_depth", "1",
    "--latent_layers", "1",
    "--latent_dim", "64",
    "--num_latents", "2",
    "--train_batch_size", "8",
    "--dev_batch_size", "8",
    "--test_batch_size", "8",
    "--n_epochs", "100",
    "--seed", "5576",
]

def get_help_text():
    try:
        p = subprocess.run(
            [PYTHON, str(MAIN), "--help"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        return p.stdout
    except Exception as e:
        return str(e)

HELP_TEXT = get_help_text()

def has_flag(flag):
    return flag in HELP_TEXT

def build_extra_args(ablation):
    """
    返回这个消融版本额外要加的参数。
    如果当前代码不支持这个消融开关，就返回 None，表示先跳过。
    """
    if ablation == "wo_quality":
        # 去掉质量感知权重
        if has_flag("--no_quality_weight"):
            return ["--no_quality_weight"]
        if has_flag("--disable_quality_weight"):
            return ["--disable_quality_weight"]
        if has_flag("--use_quality_weight"):
            return ["--use_quality_weight", "false"]
        return None

    if ablation == "wo_uncertainty":
        # 去掉退化-不确定性一致性约束
        if has_flag("--deg_uncertainty_weight"):
            return ["--deg_uncertainty_weight", "0.0"]
        return None

    if ablation == "wo_degradation":
        # 去掉多级退化建模
        # 如果你的代码有 degradation_mode 开关，这里会尝试设置为 none。
        if has_flag("--degradation_mode"):
            return ["--degradation_mode", "none"]
        if has_flag("--disable_degradation"):
            return ["--disable_degradation"]
        if has_flag("--use_degradation"):
            return ["--use_degradation", "false"]
        return None

    raise ValueError(ablation)

def parse_metrics(text):
    """
    从训练日志里抓最后一次 best 指标。
    """
    pattern = re.compile(
        r"best mae:([0-9eE+\-\.]+).*?"
        r"current corr:([0-9eE+\-\.]+).*?"
        r"acc7:([0-9eE+\-\.]+).*?"
        r"acc5:([0-9eE+\-\.]+).*?"
        r"acc2_non_zero:([0-9eE+\-\.]+).*?"
        r"f_score_non_zero:([0-9eE+\-\.]+).*?"
        r"acc2:([0-9eE+\-\.]+).*?"
        r"f_score:([0-9eE+\-\.]+)",
        re.S
    )
    matches = pattern.findall(text)
    if not matches:
        return None

    mae, corr, acc7, acc5, acc2nz, f1nz, acc2, f1 = matches[-1]
    return {
        "mae": float(mae),
        "corr": float(corr),
        "acc7": float(acc7),
        "acc5": float(acc5),
        "acc2_non_zero": float(acc2nz),
        "f1_non_zero": float(f1nz),
        "acc2": float(acc2),
        "f1": float(f1),
    }

def save_records(rows):
    df_new = pd.DataFrame(rows)

    if OUT_CSV.exists():
        df_old = pd.read_csv(OUT_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["run_id"], keep="last")
    else:
        df = df_new

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    lines = []
    lines.append("# 10 Ablation MOSI mr=0.5 seed=5576 Results")
    lines.append("")
    lines.append("这里记录 DARE-HME 三个核心模块的单 seed 消融结果。")
    lines.append("")
    lines.append("| run_id | ablation | status | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ | log_file |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")

    for _, r in df.iterrows():
        lines.append(
            f"| {r.get('run_id','')} | {r.get('ablation','')} | {r.get('status','')} | "
            f"{r.get('mae','')} | {r.get('corr','')} | {r.get('acc2','')} | {r.get('f1','')} | "
            f"{r.get('log_file','')} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

def load_existing_main_results():
    rows = []

    def read_one(csv_path, run_id, method_name):
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path)
        if "run_id" not in df.columns:
            return
        hit = df[df["run_id"].astype(str) == run_id]
        if hit.empty:
            return
        r = hit.iloc[-1]
        rows.append({
            "Method": method_name,
            "MAE ↓": float(r["mae"]),
            "Corr ↑": float(r["corr"]),
            "Acc-2 ↑": float(r["acc2"]),
            "F1 ↑": float(r["f1"]),
        })

    read_one(
        RECORDS / "02_hme_baseline_results.csv",
        "official_mosi_hme_mr05_s5576_e100",
        "HME"
    )
    read_one(
        RECORDS / "03_hme_topk_results.csv",
        "official_mosi_topk_mr05_k3_t01_s5576_e100",
        "Retrieval-only / HME-TopK"
    )
    read_one(
        RECORDS / "04_dare_hme_results.csv",
        "official_mosi_dare_mr05_s5576_e100",
        "DARE-HME full"
    )

    return rows

def generate_summary():
    rows = load_existing_main_results()

    if OUT_CSV.exists():
        df_ab = pd.read_csv(OUT_CSV)
        df_ab = df_ab[df_ab["status"] == "finished"]
        name_map = {
            "wo_quality": "DARE-HME w/o Quality",
            "wo_uncertainty": "DARE-HME w/o Uncertainty",
            "wo_degradation": "DARE-HME w/o Degradation",
        }
        for _, r in df_ab.iterrows():
            rows.append({
                "Method": name_map.get(r["ablation"], r["ablation"]),
                "MAE ↓": float(r["mae"]),
                "Corr ↑": float(r["corr"]),
                "Acc-2 ↑": float(r["acc2"]),
                "F1 ↑": float(r["f1"]),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    lines = []
    lines.append("# 10 Ablation MOSI mr=0.5 seed=5576 Summary")
    lines.append("")
    lines.append("消融实验汇总。MAE 越低越好，Corr / Acc-2 / F1 越高越好。")
    lines.append("")
    lines.append("| Method | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |")
    lines.append("|---|---:|---:|---:|---:|")

    for _, r in df.iterrows():
        lines.append(
            f"| {r['Method']} | {r['MAE ↓']:.4f} | {r['Corr ↑']:.4f} | {r['Acc-2 ↑']:.4f} | {r['F1 ↑']:.4f} |"
        )

    OUT_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pretty = []
    pretty.append("=" * 100)
    pretty.append("Ablation MOSI mr=0.5 seed=5576 Summary")
    pretty.append("=" * 100)
    pretty.append("")
    pretty.append(df.to_string(index=False))
    pretty.append("")
    pretty.append("=" * 100)
    OUT_PRETTY.write_text("\n".join(pretty), encoding="utf-8")

def run_one(ablation):
    extra = build_extra_args(ablation)
    run_id = f"ablation_mosi_{ablation}_mr05_s5576_e100"
    log_file = LOG_DIR / f"{run_id}.log"

    if extra is None:
        print("")
        print("=" * 100)
        print(f"[SKIP] {ablation}")
        print("原因：当前 HME_main_dare.py 没检测到对应命令行开关。")
        print("这不是训练错误，后面需要我带你改代码补这个消融。")
        print("=" * 100)
        return {
            "run_id": run_id,
            "ablation": ablation,
            "dataset": "MOSI",
            "missing_rate": 0.5,
            "seed": 5576,
            "epochs": 100,
            "status": "skipped_no_flag",
            "mae": "",
            "corr": "",
            "acc7": "",
            "acc5": "",
            "acc2_non_zero": "",
            "f1_non_zero": "",
            "acc2": "",
            "f1": "",
            "log_file": "",
        }

    cmd = [PYTHON] + COMMON_ARGS + extra

    print("")
    print("=" * 100)
    print(f"Run ID: {run_id}")
    print(f"Ablation: {ablation}")
    print(f"Extra args: {' '.join(extra)}")
    print(f"Log file: {log_file}")
    print("=" * 100)
    print("Command:")
    print(" ".join(cmd))
    print("=" * 100)

    with log_file.open("w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Ablation: {ablation}\n")
        f.write("Command:\n")
        f.write(" ".join(cmd) + "\n")
        f.write("=" * 100 + "\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        all_lines = []
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            all_lines.append(line)

        ret = proc.wait()

    text = "".join(all_lines)
    metrics = parse_metrics(text)

    if ret == 0 and metrics is not None:
        status = "finished"
    else:
        status = f"failed_return_{ret}"

    row = {
        "run_id": run_id,
        "ablation": ablation,
        "dataset": "MOSI",
        "missing_rate": 0.5,
        "seed": 5576,
        "epochs": 100,
        "status": status,
        "log_file": str(log_file.relative_to(ROOT)),
    }

    if metrics is None:
        row.update({
            "mae": "",
            "corr": "",
            "acc7": "",
            "acc5": "",
            "acc2_non_zero": "",
            "f1_non_zero": "",
            "acc2": "",
            "f1": "",
        })
    else:
        row.update(metrics)

    print("")
    print("=" * 100)
    print(f"Finished {run_id}, status={status}")
    if metrics:
        print(metrics)
    else:
        print("No metrics parsed. Please check log file.")
    print("=" * 100)

    return row

def main():
    print("Detected ablation-related flags in HME_main_dare.py:")
    for key in [
        "--use_quality_weight",
        "--no_quality_weight",
        "--disable_quality_weight",
        "--deg_uncertainty_weight",
        "--degradation_mode",
        "--disable_degradation",
        "--use_degradation",
    ]:
        print(f"{key}: {has_flag(key)}")

    ablations = [
        "wo_degradation",
    ]

    rows = []
    for ab in ablations:
        row = run_one(ab)
        rows.append(row)
        save_records([row])
        generate_summary()

    print("")
    print("=" * 100)
    print("Ablation first round finished.")
    print(f"Results csv: {OUT_CSV}")
    print(f"Results md: {OUT_MD}")
    print(f"Summary md: {OUT_SUMMARY_MD}")
    print(f"Pretty log: {OUT_PRETTY}")
    print("=" * 100)

if __name__ == "__main__":
    main()
