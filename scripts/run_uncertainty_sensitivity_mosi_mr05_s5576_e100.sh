#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Uncertainty weight sensitivity"
echo "Dataset: MOSI"
echo "MR: 0.5"
echo "Seed: 5576"
echo "Epochs: 100"
echo "Weights: 0.001 / 0.003 / 0.005"
echo "============================================================"

WEIGHTS=(0.001 0.003 0.005)

for W in "${WEIGHTS[@]}"
do
  WTAG=$(echo ${W} | sed 's/0\.//g')

  echo ""
  echo "############################################################"
  echo "Running deg_uncertainty_weight = ${W}"
  echo "############################################################"

  LOG_FILE="logs/uncertainty_sensitivity/mosi_mr05_s5576_e100_u${WTAG}.log"

  /opt/conda/envs/hme38/bin/python HME_MSA/HME_main_dare.py \
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
    --seed 5576 \
    --use_quality_weight true \
    --degradation_mode multi \
    --deg_uncertainty_weight ${W} \
    2>&1 | tee "${LOG_FILE}"

  echo ""
  echo "Finished weight ${W}, log saved to ${LOG_FILE}"
done

echo ""
echo "============================================================"
echo "All uncertainty sensitivity runs finished."
echo "Now generating summary..."
echo "============================================================"

python - <<'PY'
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/root/private_data/HME/HME")
LOG_DIR = ROOT / "logs" / "uncertainty_sensitivity"
RECORDS = ROOT / "records"
PRETTY = RECORDS / "pretty_logs"

OUT_CSV = RECORDS / "11_uncertainty_sensitivity_mosi_mr05_s5576.csv"
OUT_MD = RECORDS / "11_uncertainty_sensitivity_mosi_mr05_s5576.md"
OUT_PRETTY = PRETTY / "08_uncertainty_sensitivity_mosi_mr05_s5576.log"

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

rows = []

# 已有的两个点：0.0 和 0.01，从前面消融表手动加入
rows.append({
    "weight": 0.0,
    "setting": "w/o Uncertainty",
    "mae": 0.7833,
    "corr": 0.7561,
    "acc2": 0.7886,
    "f1": 0.7879,
    "log_file": "records/10_ablation_mosi_mr05_s5576_summary.md"
})

rows.append({
    "weight": 0.01,
    "setting": "DARE-HME full original",
    "mae": 0.8122,
    "corr": 0.7313,
    "acc2": 0.7828,
    "f1": 0.7832,
    "log_file": "records/04_dare_hme_results.csv"
})

for w in [0.001, 0.003, 0.005]:
    tag = str(w).replace("0.", "")
    log_file = LOG_DIR / f"mosi_mr05_s5576_e100_u{tag}.log"

    if not log_file.exists():
        rows.append({
            "weight": w,
            "setting": "sensitivity",
            "mae": "",
            "corr": "",
            "acc2": "",
            "f1": "",
            "log_file": str(log_file.relative_to(ROOT)),
        })
        continue

    text = log_file.read_text(encoding="utf-8", errors="ignore")
    matches = pattern.findall(text)

    if not matches:
        rows.append({
            "weight": w,
            "setting": "sensitivity_parse_failed",
            "mae": "",
            "corr": "",
            "acc2": "",
            "f1": "",
            "log_file": str(log_file.relative_to(ROOT)),
        })
        continue

    mae, corr, acc7, acc5, acc2nz, f1nz, acc2, f1 = matches[-1]
    rows.append({
        "weight": w,
        "setting": "sensitivity",
        "mae": float(mae),
        "corr": float(corr),
        "acc2": float(acc2),
        "f1": float(f1),
        "log_file": str(log_file.relative_to(ROOT)),
    })

df = pd.DataFrame(rows)
df = df.sort_values("weight")
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

lines = []
lines.append("# 11 Uncertainty Weight Sensitivity on MOSI mr=0.5 seed=5576")
lines.append("")
lines.append("这里比较不同 `deg_uncertainty_weight` 对 DARE-HME 的影响。")
lines.append("")
lines.append("| weight | setting | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ | log_file |")
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
        f"| {r['weight']} | {r['setting']} | {fmt(r['mae'])} | {fmt(r['corr'])} | "
        f"{fmt(r['acc2'])} | {fmt(r['f1'])} | {r['log_file']} |"
    )

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

pretty = []
pretty.append("=" * 100)
pretty.append("Uncertainty Weight Sensitivity on MOSI mr=0.5 seed=5576")
pretty.append("=" * 100)
pretty.append("")
pretty.append(df.to_string(index=False))
pretty.append("")
pretty.append("=" * 100)
OUT_PRETTY.write_text("\n".join(pretty), encoding="utf-8")

print("Generated:")
print(OUT_CSV)
print(OUT_MD)
print(OUT_PRETTY)
print("")
print(df.to_string(index=False))
PY

echo ""
echo "Summary:"
echo "records/11_uncertainty_sensitivity_mosi_mr05_s5576.md"
echo "records/pretty_logs/08_uncertainty_sensitivity_mosi_mr05_s5576.log"
