import argparse
import os
import shlex
import shutil
import pty
import select
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rate_tag(rate: float) -> str:
    s = f"{rate:.2f}".rstrip("0").rstrip(".")
    if s == "0":
        return "mr00"
    return "mr" + s.replace(".", "")


def temp_tag(temp: float) -> str:
    s = f"{temp:.3f}".rstrip("0").rstrip(".")
    return "t" + s.replace(".", "")


def model_file(model: str) -> str:
    return {
        "hme": "HME_MSA/HME_main.py",
        "topk": "HME_MSA/HME_main_topk.py",
        "dare": "HME_MSA/HME_main_dare.py",
    }[model]


def log_dir(model: str) -> Path:
    return {
        "hme": ROOT / "logs" / "hme",
        "topk": ROOT / "logs" / "hme_topk",
        "dare": ROOT / "logs" / "dare_hme",
    }[model]


def model_tag(model: str) -> str:
    return {
        "hme": "hme",
        "topk": "topk",
        "dare": "dare",
    }[model]


def build_run_id(args) -> str:
    ds = args.dataset.lower()
    mr = rate_tag(args.missing_rate)

    if args.model == "topk":
        return (
            f"{args.tag}_{ds}_topk_{mr}_k{args.top_k}_"
            f"{temp_tag(args.temperature)}_s{args.seed}_e{args.epochs}"
        )

    return f"{args.tag}_{ds}_{model_tag(args.model)}_{mr}_s{args.seed}_e{args.epochs}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["hme", "topk", "dare"], required=True)
    parser.add_argument("--dataset", default="mosi")
    parser.add_argument("--missing_rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=5576)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tag", default="sanity")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--note", default="")

    parser.add_argument("--learning_rate", default="2e-5")
    parser.add_argument("--d_l", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hyper_depth", type=int, default=1)
    parser.add_argument("--latent_layers", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_latents", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--dev_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)

    # Modality feature dimensions.
    # MOSI: visual=20, acoustic=5
    # MOSEI: visual=35, acoustic=74
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--visual_dim", type=int, default=None)
    parser.add_argument("--acoustic_dim", type=int, default=None)

    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    # Auto set modality dimensions according to dataset
    if args.visual_dim is None or args.acoustic_dim is None:
        if args.dataset.lower() == "mosei":
            args.visual_dim = 35
            args.acoustic_dim = 74
        else:
            args.visual_dim = 20
            args.acoustic_dim = 5

    run_id = args.run_id or build_run_id(args)

    this_log_dir = log_dir(args.model)
    this_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = this_log_dir / f"{run_id}.log"

    cmd = [
        sys.executable,
        model_file(args.model),
        "--dataset", args.dataset,
        "--learning_rate", str(args.learning_rate),
        "--d_l", str(args.d_l),
        "--missing_rate", str(args.missing_rate),
        "--layers", str(args.layers),
        "--hyper_depth", str(args.hyper_depth),
        "--latent_layers", str(args.latent_layers),
        "--latent_dim", str(args.latent_dim),
        "--num_latents", str(args.num_latents),
        "--train_batch_size", str(args.train_batch_size),
        "--dev_batch_size", str(args.dev_batch_size),
        "--test_batch_size", str(args.test_batch_size),
        "--n_epochs", str(args.epochs),
        "--seed", str(args.seed),
    ]

    if args.model == "topk":
        cmd.extend([
            "--top_k", str(args.top_k),
            "--top_temperature", str(args.temperature),
        ])

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    print("=" * 80)
    print("Run ID:", run_id)
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Missing rate:", args.missing_rate)
    print("Seed:", args.seed)
    print("Epochs:", args.epochs)
    print("Log file:", log_file.relative_to(ROOT))
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)

    if args.dry_run:
        print("Dry run only. No experiment executed.")
        return

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 先写入实验头信息
    with log_file.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"run_id: {run_id}\n")
        f.write(f"start_time: {start_time}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"missing_rate: {args.missing_rate}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write("command: " + " ".join(cmd) + "\n")
        f.write("=" * 80 + "\n\n")

    # 用 script 命令分配伪终端：
    # 1. tqdm 会认为自己在真实终端里，所以能单行动态刷新
    # 2. 同时把完整输出追加保存到 log 文件
    cmd_str = shlex.join(cmd)
    script_bin = shutil.which("script")

    if script_bin is not None:
        script_cmd = [
            script_bin,
            "-q",
            "-f",
            "-a",
            "-c",
            cmd_str,
            str(log_file),
        ]
        process = subprocess.run(
            script_cmd,
            cwd=str(ROOT),
            env=env,
        )
        return_code = process.returncode
    else:
        print("Warning: script command not found. Fallback to direct terminal output without full log capture.")
        process = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
        )
        return_code = process.returncode

    status = "finished" if return_code == 0 else "failed"

    record_cmd = [
        sys.executable,
        str(ROOT / "tools" / "record_result.py"),
        "--run_id", run_id,
        "--model", args.model,
        "--dataset", args.dataset,
        "--missing_rate", str(args.missing_rate),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--status", status,
        "--log_file", str(log_file),
        "--note", args.note,
    ]

    if args.model == "topk":
        record_cmd.extend([
            "--top_k", str(args.top_k),
            "--temperature", str(args.temperature),
        ])

    if args.model == "dare":
        record_cmd.extend([
            "--degradation_mode", "multi-level",
            "--quality_module", "on",
            "--uncertainty_module", "on",
        ])

    print("\n" + "=" * 80)
    print("Recording result...")
    subprocess.run(record_cmd, cwd=str(ROOT), check=False)

    if return_code != 0:
        print(f"Experiment failed. Return code: {return_code}")
        sys.exit(return_code)

    print("Experiment finished and recorded.")


if __name__ == "__main__":
    main()
