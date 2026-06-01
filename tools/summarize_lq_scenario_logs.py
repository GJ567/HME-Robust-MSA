from pathlib import Path
import re
import pandas as pd

ROOT = Path("/root/private_data/HME/HME")
LOG_DIR = ROOT / "records" / "27_lq_scenario_logs"

SCENARIOS = ["audio", "vision", "text", "audio_vision"]
METHODS = {
    "HME": "hme",
    "DARE-HME": "dare",
}

pattern_best = re.compile(
    r"best mae:(?P<mae>[-+]?\d*\.\d+).*?"
    r"current corr:(?P<corr>[-+]?\d*\.\d+).*?"
    r"acc2_non_zero:(?P<acc2_nz>[-+]?\d*\.\d+).*?"
    r"f_score_non_zero:(?P<f1_nz>[-+]?\d*\.\d+).*?"
    r"acc2:(?P<acc2>[-+]?\d*\.\d+).*?"
    r"f_score:(?P<f1>[-+]?\d*\.\d+)",
    re.S,
)

pattern_current = re.compile(
    r"current mae:(?P<mae>[-+]?\d*\.\d+).*?"
    r"current corr:(?P<corr>[-+]?\d*\.\d+).*?"
    r"acc2_non_zero:(?P<acc2_nz>[-+]?\d*\.\d+).*?"
    r"f_score_non_zero:(?P<f1_nz>[-+]?\d*\.\d+).*?"
    r"acc2:(?P<acc2>[-+]?\d*\.\d+).*?"
    r"f_score:(?P<f1>[-+]?\d*\.\d+)",
    re.S,
)

def parse_one(path):
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(pattern_best.finditer(text))

    if not matches:
        matches = list(pattern_current.finditer(text))

    if not matches:
        return None

    m = matches[-1].groupdict()
    return {k: float(v) for k, v in m.items()}

rows = []

for scenario in SCENARIOS:
    for method_name, prefix in METHODS.items():
        log_path = LOG_DIR / f"{prefix}_{scenario}_s5576.log"
        res = parse_one(log_path)

        if res is None:
            rows.append({
                "scenario": scenario,
                "method": method_name,
                "seed": 5576,
                "mae": None,
                "corr": None,
                "acc2": None,
                "f1": None,
                "acc2_non_zero": None,
                "f1_non_zero": None,
                "log": str(log_path),
                "status": "missing_or_parse_failed",
            })
        else:
            rows.append({
                "scenario": scenario,
                "method": method_name,
                "seed": 5576,
                "mae": res["mae"],
                "corr": res["corr"],
                "acc2": res["acc2"],
                "f1": res["f1"],
                "acc2_non_zero": res["acc2_nz"],
                "f1_non_zero": res["f1_nz"],
                "log": str(log_path),
                "status": "ok",
            })

df = pd.DataFrame(rows)

out_csv = ROOT / "records" / "27_lq_scenario_mosi_mr05_s5576_summary.csv"
out_md = ROOT / "records" / "27_lq_scenario_mosi_mr05_s5576_summary.md"

df.to_csv(out_csv, index=False)

md = []
md.append("# 27 Low-quality Scenario Extension on MOSI mr=0.5 seed=5576")
md.append("")
md.append("该实验用于验证 DARE-HME 是否不仅在整体 random missing 设置下有效，也能在不同模态来源的低质量场景下保持鲁棒性。")
md.append("")
md.append("当前版本采用 binary severe degradation：")
md.append("")
md.append("- 对 HME：对应模态随机置为 missing。")
md.append("- 对 DARE-HME：对应模态随机设为 degradation level 4，即 complete missing。")
md.append("- 未指定退化的模态保持完整。")
md.append("")
md.append("## Raw Results")
md.append("")
md.append("| Scenario | Method | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ | Acc-2 Non0 ↑ | F1 Non0 ↑ | Status |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

for _, r in df.iterrows():
    def fmt(x):
        if pd.isna(x):
            return "-"
        return f"{x:.4f}"
    md.append(
        f"| {r['scenario']} | {r['method']} | {fmt(r['mae'])} | {fmt(r['corr'])} | "
        f"{fmt(r['acc2'])} | {fmt(r['f1'])} | {fmt(r['acc2_non_zero'])} | {fmt(r['f1_non_zero'])} | {r['status']} |"
    )

md.append("")
md.append("## DARE-HME Improvement over HME")
md.append("")
md.append("| Scenario | ΔMAE ↓ | ΔCorr ↑ | ΔAcc-2 ↑ | ΔF1 ↑ |")
md.append("|---|---:|---:|---:|---:|")

for scenario in SCENARIOS:
    sub = df[(df["scenario"] == scenario) & (df["status"] == "ok")]
    if set(sub["method"]) >= {"HME", "DARE-HME"}:
        h = sub[sub["method"] == "HME"].iloc[0]
        d = sub[sub["method"] == "DARE-HME"].iloc[0]
        delta_mae = h["mae"] - d["mae"]
        delta_corr = d["corr"] - h["corr"]
        delta_acc2 = d["acc2"] - h["acc2"]
        delta_f1 = d["f1"] - h["f1"]
        md.append(
            f"| {scenario} | {delta_mae:+.4f} | {delta_corr:+.4f} | {delta_acc2:+.4f} | {delta_f1:+.4f} |"
        )
    else:
        md.append(f"| {scenario} | - | - | - | - |")

md.append("")
md.append("## How to use this table in paper")
md.append("")
md.append("This table can be used as a stress-test analysis to show that DARE-HME remains robust when the low-quality source is concentrated on a specific modality rather than uniformly distributed over all modalities.")

out_md.write_text("\n".join(md), encoding="utf-8")

print("written:", out_md)
print("written:", out_csv)
print("")
print("\n".join(md))
