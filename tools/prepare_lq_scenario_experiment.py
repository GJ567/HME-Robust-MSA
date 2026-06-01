from pathlib import Path
import shutil
import re

ROOT = Path("/root/private_data/HME/HME")
HME_DIR = ROOT / "HME_MSA"

def patch_hme():
    src = HME_DIR / "HME_main.py"
    dst = HME_DIR / "HME_main_lq.py"

    if not src.exists():
        raise FileNotFoundError(src)

    shutil.copy2(src, dst)
    text = dst.read_text(encoding="utf-8", errors="ignore")

    # 1. 添加 lq_scenario 参数
    if "--lq_scenario" not in text:
        text = re.sub(
            r'(parser\.add_argument\("--missing_rate".*?\n)',
            r'\1parser.add_argument("--lq_scenario", type=str, default="all", '
            r'choices=["all", "text", "audio", "vision", "text_audio", "text_vision", "audio_vision"], '
            r'help="low-quality scenario: which modality is selectively missing/degraded")\n',
            text,
            count=1,
        )

    helper = r'''
def generate_lq_missing_matrix(nums, num_modalities, missing_rate=0.5, scenario="all"):
    """
    Generate modality-specific binary missing matrix for low-quality scenario stress test.

    Matrix meaning for HME:
      1 = modality available
      0 = modality missing

    scenario:
      all          : original random missing over all modalities
      text         : only text may be missing
      audio        : only audio may be missing
      vision       : only vision may be missing
      text_audio   : text and audio may be missing
      text_vision  : text and vision may be missing
      audio_vision : audio and vision may be missing
    """
    scenario = str(scenario).lower()

    if scenario == "all":
        return generate_missing_matrix(nums, num_modalities, missing_rate=missing_rate)

    scenario_map = {
        "text": [0],
        "audio": [1],
        "vision": [2],
        "text_audio": [0, 1],
        "text_vision": [0, 2],
        "audio_vision": [1, 2],
    }

    if scenario not in scenario_map:
        raise ValueError(f"Unknown lq_scenario: {scenario}")

    ps = np.ones((nums, num_modalities), dtype=np.int32)
    affected = scenario_map[scenario]

    for m in affected:
        miss = (np.random.rand(nums) < float(missing_rate)).astype(np.int32)
        ps[:, m] = 1 - miss

    return ps
'''

    if "def generate_lq_missing_matrix" not in text:
        text = text.replace("\ndef train_epoch(", "\n" + helper + "\ndef train_epoch(", 1)

    text = text.replace(
        "generate_missing_matrix(nums, 3, missing_rate=args.missing_rate)",
        "generate_lq_missing_matrix(nums, 3, missing_rate=args.missing_rate, scenario=args.lq_scenario)"
    )

    # 记录 scenario 到日志
    if 'f.write("LQ scenario:' not in text and 'f.write("Missing rate:' in text:
        text = text.replace(
            '        f.write("Missing rate: {}\\n".format(args.missing_rate))',
            '        f.write("Missing rate: {}\\n".format(args.missing_rate))\n'
            '        f.write("LQ scenario: {}\\n".format(args.lq_scenario))'
        )

    dst.write_text(text, encoding="utf-8")
    print("patched:", dst)


def patch_dare():
    src = HME_DIR / "HME_main_dare.py"
    dst = HME_DIR / "HME_main_dare_lq.py"

    if not src.exists():
        raise FileNotFoundError(src)

    shutil.copy2(src, dst)
    text = dst.read_text(encoding="utf-8", errors="ignore")

    # 1. 添加 lq_scenario 和 lq_level_mode 参数
    if "--lq_scenario" not in text:
        text = re.sub(
            r'(parser\.add_argument\("--missing_rate".*?\n)',
            r'\1parser.add_argument("--lq_scenario", type=str, default="all", '
            r'choices=["all", "text", "audio", "vision", "text_audio", "text_vision", "audio_vision"], '
            r'help="low-quality scenario: which modality is selectively degraded")\n'
            r'parser.add_argument("--lq_level_mode", type=str, default="binary", '
            r'choices=["binary", "multi"], '
            r'help="binary uses level 0/4 only; multi uses levels 0-4")\n',
            text,
            count=1,
        )

    helper = r'''
def generate_lq_degradation_matrix(nums, num_modalities, degradation_rate=0.5, scenario="all", lq_level_mode="binary"):
    """
    Generate modality-specific degradation matrix for DARE-HME low-quality scenario stress test.

    Matrix meaning for DARE-HME:
      0 = complete / clean
      1 = slight degradation
      2 = moderate degradation
      3 = severe degradation
      4 = complete missing

    For the first fair stress test, we use lq_level_mode="binary":
      only level 0 and level 4 are used, so HME and DARE-HME face comparable severe missing patterns.

    Later, lq_level_mode="multi" can be used to test real multi-level low-quality degradation.
    """
    scenario = str(scenario).lower()
    lq_level_mode = str(lq_level_mode).lower()

    scenario_map = {
        "all": [0, 1, 2],
        "text": [0],
        "audio": [1],
        "vision": [2],
        "text_audio": [0, 1],
        "text_vision": [0, 2],
        "audio_vision": [1, 2],
    }

    if scenario not in scenario_map:
        raise ValueError(f"Unknown lq_scenario: {scenario}")

    if scenario == "all" and lq_level_mode == "multi":
        return generate_degradation_matrix(nums, num_modalities, degradation_rate=degradation_rate)

    mat = np.zeros((nums, num_modalities), dtype=np.int64)
    affected = scenario_map[scenario]

    if lq_level_mode == "binary":
        # level 4 means complete missing; level 0 means clean.
        for m in affected:
            miss = (np.random.rand(nums) < float(degradation_rate)).astype(np.int64)
            mat[:, m] = miss * 4

        # If all three modalities are allowed to degrade, avoid all-missing samples.
        if scenario == "all":
            for i in range(nums):
                if np.all(mat[i] == 4):
                    keep_m = np.random.randint(0, num_modalities)
                    mat[i, keep_m] = 0

        return mat

    elif lq_level_mode == "multi":
        # Use the original multi-level degradation distribution only on selected modalities.
        for m in affected:
            mat[:, m] = generate_degradation_matrix(nums, 1, degradation_rate=degradation_rate).reshape(-1)
        return mat

    else:
        raise ValueError(f"Unknown lq_level_mode: {lq_level_mode}")
'''

    if "def generate_lq_degradation_matrix" not in text:
        text = text.replace("\ndef print_degradation_statistics(", "\n" + helper + "\ndef print_degradation_statistics(", 1)

    text = text.replace(
        "generate_degradation_matrix(nums, 3, degradation_rate=args.missing_rate)",
        "generate_lq_degradation_matrix(nums, 3, degradation_rate=args.missing_rate, scenario=args.lq_scenario, lq_level_mode=args.lq_level_mode)"
    )

    # 记录 scenario 到日志
    if 'f.write("LQ scenario:' not in text and 'f.write("Missing rate:' in text:
        text = text.replace(
            '        f.write("Missing rate: {}\\n".format(args.missing_rate))',
            '        f.write("Missing rate: {}\\n".format(args.missing_rate))\n'
            '        f.write("LQ scenario: {}\\n".format(args.lq_scenario))\n'
            '        f.write("LQ level mode: {}\\n".format(args.lq_level_mode))'
        )

    dst.write_text(text, encoding="utf-8")
    print("patched:", dst)


patch_hme()
patch_dare()

print("\nCheck:")
print("  HME_MSA/HME_main_lq.py")
print("  HME_MSA/HME_main_dare_lq.py")
