#!/bin/bash

cd /root/private_data/HME/HME

echo "============================================================"
echo "Official DARE-HME u=0.001 on MOSEI"
echo "Missing rates: 0.2 / 0.5"
echo "Seed: 5576"
echo "Epochs: 100"
echo "Purpose: make MOSEI generalization consistent with final MOSI setting"
echo "============================================================"

# HME_main_dare.py 一般支持 MOSEI 的特征维度参数。
# 如果当前代码不支持这些参数，就自动不加。
DIM_ARGS=""
if /opt/conda/envs/hme38/bin/python HME_MSA/HME_main_dare.py --help 2>&1 | grep -q -- "--ACOUSTIC_DIM"; then
  DIM_ARGS="--TEXT_DIM 768 --VISUAL_DIM 35 --ACOUSTIC_DIM 74"
fi

echo "DIM_ARGS=${DIM_ARGS}"

MRS=(0.2 0.5)

for MR in "${MRS[@]}"
do
  if [ "$MR" = "0.2" ]; then
    MRTAG="mr02"
  elif [ "$MR" = "0.5" ]; then
    MRTAG="mr05"
  else
    MRTAG="mrxx"
  fi

  echo ""
  echo "############################################################"
  echo "Running DARE-HME u=0.001, MOSEI ${MRTAG}, seed=5576"
  echo "############################################################"

  LOG_FILE="logs/official_u001/dare_u001_mosei_${MRTAG}_s5576_e100.log"

  script -q -c "/opt/conda/envs/hme38/bin/python HME_MSA/HME_main_dare.py \
    --dataset mosei \
    ${DIM_ARGS} \
    --learning_rate 2e-5 \
    --d_l 64 \
    --missing_rate ${MR} \
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
    --deg_uncertainty_weight 0.001" \
    "${LOG_FILE}"

  echo ""
  echo "Finished MOSEI ${MRTAG}, log saved to ${LOG_FILE}"
done

echo ""
echo "============================================================"
echo "Generating final MOSEI u=0.001 summary..."
echo "============================================================"

python - <<'PY'
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/root/private_data/HME/HME")
RECORDS = ROOT / "records"
PRETTY = RECORDS / "pretty_logs"
LOG_DIR = ROOT / "logs" / "official_u001"

OUT_CSV = RECORDS / "15_final_mosei_hme_vs_dare_u001_summary.csv"
OUT_MD = RECORDS / "15_final_mosei_hme_vs_dare_u001_summary.md"
OUT_IMPROVE_CSV = RECORDS / "15_final_mosei_hme_vs_dare_u001_improvement.csv"
OUT_PRETTY = PRETTY / "10_final_mosei_hme_vs_dare_u001_summary.log"

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

def parse_log(mr, log_file):
    if not log_file.exists():
        return None

    text = log_file.read_text(encoding="utf-8", errors="ignore")
    matches = pattern.findall(text)

    if not matches:
        return None

    mae, corr, acc7, acc5, acc2nz, f1nz, acc2, f1 = matches[-1]
    return {
        "dataset": "MOSEI",
        "missing_rate": mr,
        "model": "DARE-HME u=0.001",
        "seed": 5576,
        "mae": float(mae),
        "corr": float(corr),
        "acc2": float(acc2),
        "f1": float(f1),
        "source": str(log_file.relative_to(ROOT)),
    }

def read_hme_from_records():
    path = RECORDS / "02_hme_baseline_results.csv"
    rows = []

    if not path.exists():
        return rows

    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].astype(str).str.upper()
    df["run_id"] = df["run_id"].astype(str)

    for mr, tag in [(0.2, "mr02"), (0.5, "mr05")]:
        hit = df[
            (df["dataset"] == "MOSEI") &
            (df["run_id"].str.contains(f"official_mosei_hme_{tag}_s5576_e100", regex=False))
        ]

        if hit.empty:
            # 兼容另一种 run_id 写法
            hit = df[
                (df["dataset"] == "MOSEI") &
                (pd.to_numeric(df["missing_rate"], errors="coerce") == mr) &
                (pd.to_numeric(df["seed"], errors="coerce") == 5576) &
                (pd.to_numeric(df["epochs"], errors="coerce") == 100)
            ]

        if not hit.empty:
            r = hit.iloc[-1]
            rows.append({
                "dataset": "MOSEI",
                "missing_rate": mr,
                "model": "HME",
                "seed": 5576,
                "mae": float(r["mae"]),
                "corr": float(r["corr"]),
                "acc2": float(r["acc2"]),
                "f1": float(r["f1"]),
                "source": "records/02_hme_baseline_results.csv",
            })

    return rows

rows = []
rows.extend(read_hme_from_records())

for mr, tag in [(0.2, "mr02"), (0.5, "mr05")]:
    parsed = parse_log(mr, LOG_DIR / f"dare_u001_mosei_{tag}_s5576_e100.log")
    if parsed:
        rows.append(parsed)

df = pd.DataFrame(rows)

if df.empty:
    print("No results parsed.")
    raise SystemExit(1)

df = df.sort_values(["missing_rate", "model"])
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

# Markdown 主表
lines = []
lines.append("# 15 Final MOSEI HME vs DARE-HME u=0.001 Summary")
lines.append("")
lines.append("MOSEI 泛化验证表。DARE-HME 使用最终配置 `deg_uncertainty_weight=0.001`。")
lines.append("")
lines.append("| Dataset | MR | Model | Seed | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |")
lines.append("|---|---:|---|---:|---:|---:|---:|---:|")

for _, r in df.iterrows():
    lines.append(
        f"| {r['dataset']} | {r['missing_rate']} | {r['model']} | {int(r['seed'])} | "
        f"{r['mae']:.4f} | {r['corr']:.4f} | {r['acc2']:.4f} | {r['f1']:.4f} |"
    )

# 提升表
improve_rows = []
for mr in [0.2, 0.5]:
    part = df[df["missing_rate"] == mr].set_index("model")
    if "HME" in part.index and "DARE-HME u=0.001" in part.index:
        h = part.loc["HME"]
        d = part.loc["DARE-HME u=0.001"]
        improve_rows.append({
            "dataset": "MOSEI",
            "missing_rate": mr,
            "comparison": "DARE-HME u=0.001 vs HME",
            "mae_reduction": round(h["mae"] - d["mae"], 4),
            "corr_gain": round(d["corr"] - h["corr"], 4),
            "acc2_gain": round(d["acc2"] - h["acc2"], 4),
            "acc2_gain_percent_point": round((d["acc2"] - h["acc2"]) * 100, 2),
            "f1_gain": round(d["f1"] - h["f1"], 4),
            "f1_gain_percent_point": round((d["f1"] - h["f1"]) * 100, 2),
        })

improve = pd.DataFrame(improve_rows)
improve.to_csv(OUT_IMPROVE_CSV, index=False, encoding="utf-8")

if not improve.empty:
    lines.append("")
    lines.append("## Improvement")
    lines.append("")
    lines.append("| MR | MAE reduction | Corr gain | Acc-2 gain | F1 gain |")
    lines.append("|---:|---:|---:|---:|---:|")
    for _, r in improve.iterrows():
        lines.append(
            f"| {r['missing_rate']} | {r['mae_reduction']:.4f} | {r['corr_gain']:.4f} | "
            f"+{r['acc2_gain_percent_point']:.2f} pp | +{r['f1_gain_percent_point']:.2f} pp |"
        )

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

pretty = []
pretty.append("=" * 100)
pretty.append("Final MOSEI HME vs DARE-HME u=0.001 Summary")
pretty.append("=" * 100)
pretty.append("")
pretty.append(df.to_string(index=False))
pretty.append("")
if not improve.empty:
    pretty.append("Improvement:")
    pretty.append(improve.to_string(index=False))
pretty.append("")
pretty.append("=" * 100)

OUT_PRETTY.write_text("\n".join(pretty), encoding="utf-8")

print("Generated:")
print(OUT_CSV)
print(OUT_MD)
print(OUT_IMPROVE_CSV)
print(OUT_PRETTY)
print("")
print(df.to_string(index=False))
if not improve.empty:
    print("")
    print("Improvement:")
    print(improve.to_string(index=False))
PY

echo ""
echo "Done."
echo "Check:"
echo "records/15_final_mosei_hme_vs_dare_u001_summary.md"
echo "records/15_final_mosei_hme_vs_dare_u001_improvement.csv"
echo "records/pretty_logs/10_final_mosei_hme_vs_dare_u001_summary.log"
