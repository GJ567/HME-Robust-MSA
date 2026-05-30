#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Official DARE-HME u=0.001 on MOSI mr=0.5"
echo "Seeds: 1111 / 2222"
echo "Epochs: 100"
echo "Purpose: replace old DARE-HME full weight=0.01"
echo "============================================================"

SEEDS=(1111 2222)

for SEED in "${SEEDS[@]}"
do
  echo ""
  echo "############################################################"
  echo "Running DARE-HME u=0.001, MOSI mr=0.5, seed=${SEED}"
  echo "############################################################"

  LOG_FILE="logs/official_u001/dare_u001_mosi_mr05_s${SEED}_e100.log"

  script -q -c "/opt/conda/envs/hme38/bin/python HME_MSA/HME_main_dare.py \
    --dataset mosi \
    --learning_rate 2e-5 \
    --d_l 64 \
    --missing_rate 0.5 \
    --layers 1 \
    --hyper_depth 1 \
    --latent_layers 1 \
    --latent_dim 64 \
    --num_latents 2 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --n_epochs 100 \
    --seed ${SEED} \
    --use_quality_weight true \
    --degradation_mode multi \
    --deg_uncertainty_weight 0.001" \
    "${LOG_FILE}"

  echo ""
  echo "Finished seed=${SEED}, log saved to ${LOG_FILE}"
done

echo ""
echo "============================================================"
echo "Generating mr=0.5 u=0.001 3-seed summary..."
echo "============================================================"

python - <<'PY'
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/root/private_data/HME/HME")
LOG_DIR = ROOT / "logs" / "official_u001"
OLD_U001_LOG = ROOT / "logs" / "uncertainty_sensitivity" / "mosi_mr05_s5576_e100_u001.log"

RECORDS = ROOT / "records"
PRETTY = RECORDS / "pretty_logs"

OUT_CSV = RECORDS / "13_dare_u001_mosi_mr05_3seed_results.csv"
OUT_MD = RECORDS / "13_dare_u001_mosi_mr05_3seed_summary.md"
OUT_PRETTY = PRETTY / "09_dare_u001_mosi_mr05_3seed_summary.log"

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

def parse_log(seed, log_file):
    if not log_file.exists():
        return {
            "seed": seed,
            "status": "missing_log",
            "mae": "",
            "corr": "",
            "acc2": "",
            "f1": "",
            "log_file": str(log_file.relative_to(ROOT)) if log_file.is_absolute() else str(log_file),
        }

    text = log_file.read_text(encoding="utf-8", errors="ignore")
    matches = pattern.findall(text)

    if not matches:
        return {
            "seed": seed,
            "status": "parse_failed",
            "mae": "",
            "corr": "",
            "acc2": "",
            "f1": "",
            "log_file": str(log_file.relative_to(ROOT)),
        }

    mae, corr, acc7, acc5, acc2nz, f1nz, acc2, f1 = matches[-1]
    return {
        "seed": seed,
        "status": "finished",
        "mae": float(mae),
        "corr": float(corr),
        "acc2": float(acc2),
        "f1": float(f1),
        "log_file": str(log_file.relative_to(ROOT)),
    }

rows = []

# seed=5576 已经在 uncertainty sensitivity 中跑过，作为 u=0.001 full 结果
rows.append(parse_log(5576, OLD_U001_LOG))

for seed in [1111, 2222]:
    rows.append(parse_log(seed, LOG_DIR / f"dare_u001_mosi_mr05_s{seed}_e100.log"))

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

valid = df[df["status"] == "finished"].copy()

lines = []
lines.append("# 13 DARE-HME u=0.001 MOSI mr=0.5 3-seed Summary")
lines.append("")
lines.append("新版 DARE-HME full 使用 `deg_uncertainty_weight=0.001`。")
lines.append("")
lines.append("| seed | status | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ | log_file |")
lines.append("|---:|---|---:|---:|---:|---:|---|")

for _, r in df.iterrows():
    def fmt(x):
        if x == "":
            return ""
        try:
            return f"{float(x):.4f}"
        except:
            return str(x)

    lines.append(
        f"| {r['seed']} | {r['status']} | {fmt(r['mae'])} | {fmt(r['corr'])} | "
        f"{fmt(r['acc2'])} | {fmt(r['f1'])} | {r['log_file']} |"
    )

if len(valid) > 0:
    lines.append("")
    lines.append("## Mean ± Std")
    lines.append("")
    lines.append("| Model | Seeds | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |")
    lines.append("|---|---|---:|---:|---:|---:|")

    seeds = "/".join(map(str, sorted(valid["seed"].astype(int).tolist())))

    def ms(col):
        return f"{valid[col].mean():.4f}±{valid[col].std():.4f}"

    lines.append(
        f"| DARE-HME u=0.001 | {seeds} | {ms('mae')} | {ms('corr')} | {ms('acc2')} | {ms('f1')} |"
    )

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

pretty = []
pretty.append("=" * 100)
pretty.append("DARE-HME u=0.001 MOSI mr=0.5 3-seed Summary")
pretty.append("=" * 100)
pretty.append("")
pretty.append(df.to_string(index=False))
pretty.append("")
if len(valid) > 0:
    pretty.append("Mean ± Std:")
    pretty.append(f"MAE  = {valid['mae'].mean():.4f}±{valid['mae'].std():.4f}")
    pretty.append(f"Corr = {valid['corr'].mean():.4f}±{valid['corr'].std():.4f}")
    pretty.append(f"Acc2 = {valid['acc2'].mean():.4f}±{valid['acc2'].std():.4f}")
    pretty.append(f"F1   = {valid['f1'].mean():.4f}±{valid['f1'].std():.4f}")
pretty.append("")
pretty.append("=" * 100)

OUT_PRETTY.write_text("\n".join(pretty), encoding="utf-8")

print("Generated:")
print(OUT_CSV)
print(OUT_MD)
print(OUT_PRETTY)
print("")
print(df.to_string(index=False))
if len(valid) > 0:
    print("")
    print("Mean ± Std:")
    print(f"MAE  = {valid['mae'].mean():.4f}±{valid['mae'].std():.4f}")
    print(f"Corr = {valid['corr'].mean():.4f}±{valid['corr'].std():.4f}")
    print(f"Acc2 = {valid['acc2'].mean():.4f}±{valid['acc2'].std():.4f}")
    print(f"F1   = {valid['f1'].mean():.4f}±{valid['f1'].std():.4f}")
PY

echo ""
echo "Done."
echo "Check:"
echo "records/13_dare_u001_mosi_mr05_3seed_summary.md"
echo "records/pretty_logs/09_dare_u001_mosi_mr05_3seed_summary.log"
