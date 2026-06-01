from pathlib import Path
import re
import shutil

path = Path("HME_MSA/HME_main_dare_lq.py")
bak = Path("HME_MSA/HME_main_dare_lq.py.bak_fix_lq_call_v2")

if not path.exists():
    raise FileNotFoundError(path)

if not bak.exists():
    shutil.copy2(path, bak)
    print("backup:", bak)

text = path.read_text(encoding="utf-8", errors="ignore")

old = r"ps\s*=\s*generate_degradation_matrix\(\s*nums\s*,\s*3\s*,\s*degradation_rate\s*=\s*args\.missing_rate\s*,\s*degradation_mode\s*=\s*args\.degradation_mode\s*\)"

new = (
    "ps = generate_lq_degradation_matrix("
    "nums, 3, degradation_rate=args.missing_rate, "
    "scenario=args.lq_scenario, lq_level_mode=args.lq_level_mode)"
)

text2, n = re.subn(old, new, text)

path.write_text(text2, encoding="utf-8")

print("changed calls:", n)
print("")
print("generate_lq_degradation_matrix occurrences:")
for i, line in enumerate(text2.splitlines(), 1):
    if "generate_lq_degradation_matrix" in line:
        print(f"{i}: {line}")

print("")
print("remaining generate_degradation_matrix occurrences:")
for i, line in enumerate(text2.splitlines(), 1):
    if "generate_degradation_matrix" in line:
        print(f"{i}: {line}")
