# -*- coding: utf-8 -*-
"""
benchmark/run_full_benchmark.py
Master pipeline: train → generate → evaluate for DaiViet-Pattern Benchmark.

Steps:
  1. Train M2, M3, M6 sequentially (each ~hours on RTX 4080)
  2. Generate images for M1, M2, M3, M6 (50 imgs × 3 periods each)
  3. Evaluate all models (FID, CLIP, SSIM, LPIPS, PSNR)

Usage:
  # Full pipeline
  python benchmark/run_full_benchmark.py

  # Skip training (already done), run generate + evaluate
  python benchmark/run_full_benchmark.py --skip_train

  # Specific models only
  python benchmark/run_full_benchmark.py --models M2,M6
  python benchmark/run_full_benchmark.py --skip_train --models M2,M6
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
BENCHMARK_PY = REPO_ROOT / "benchmark"

# Rough time estimates (RTX 4080 16 GB)
# (train_hours, generate_minutes, eval_minutes)
TIME_ESTIMATES = {
    "M1": (0.0,   15, 20),   # no training
    "M2": (6.0,   15, 20),   # SDXL LoRA 50ep
    "M3": (2.5,    8, 20),   # SD1.5 LoRA 50ep
    "M6": (10.0,  15, 20),   # SDXL LoRA + cultural loss 50ep
}

ALL_MODELS  = ["M1", "M2", "M3", "M6"]
TRAIN_MODELS = ["M2", "M3", "M6"]   # M1 has no training


# ─────────────────────────────────────────────────────────────────────────────
def _run(script: Path, extra_args: list[str], step_label: str) -> bool:
    """Run a benchmark sub-script. Returns True on success, False on failure."""
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n  CMD: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        elapsed = (time.time() - t0) / 60
        print(f"  {step_label}: DONE in {elapsed:.1f} min")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = (time.time() - t0) / 60
        print(f"\n  [FAILED] {step_label} after {elapsed:.1f} min "
              f"(exit code {e.returncode})", file=sys.stderr)
        return False
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] {step_label}", file=sys.stderr)
        raise


# ─────────────────────────────────────────────────────────────────────────────
def estimate_total_time(models: list[str], skip_train: bool):
    """Print estimated time before starting."""
    total_train_h   = 0.0
    total_gen_min   = 0.0
    total_eval_min  = 0.0

    for m in models:
        train_h, gen_min, eval_min = TIME_ESTIMATES.get(m, (0, 15, 20))
        if not skip_train and m in TRAIN_MODELS:
            total_train_h  += train_h
        total_gen_min  += gen_min
        total_eval_min += eval_min

    total_h = total_train_h + total_gen_min / 60 + total_eval_min / 60

    print("\n" + "=" * 60)
    print("  ESTIMATED TIME (RTX 4080 16 GB)")
    print("=" * 60)
    print(f"  Models          : {', '.join(models)}")
    if not skip_train:
        print(f"  Training        : ~{total_train_h:.1f} h")
    else:
        print(f"  Training        : SKIPPED")
    print(f"  Generation      : ~{total_gen_min:.0f} min")
    print(f"  Evaluation      : ~{total_eval_min:.0f} min")
    print(f"  Total estimate  : ~{total_h:.1f} h")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def run_training(models: list[str]) -> dict[str, bool]:
    """Train all trainable models sequentially. Returns {model: success}."""
    results: dict[str, bool] = {}
    train_script = BENCHMARK_PY / "train_benchmark.py"

    for m in models:
        if m not in TRAIN_MODELS:
            print(f"\n  {m}: no training needed (vanilla baseline)")
            results[m] = True
            continue

        print(f"\n{'='*60}")
        print(f"  TRAINING {m}")
        print(f"{'='*60}")
        ok = _run(train_script, ["--model", m], f"train {m}")
        results[m] = ok

        if not ok:
            print(f"  [WARN] {m} training failed — generation/eval will be skipped",
                  file=sys.stderr)

    return results


# ─────────────────────────────────────────────────────────────────────────────
def run_generation(models: list[str],
                   train_results: dict[str, bool]) -> dict[str, bool]:
    """Generate images for each model. Skips models whose training failed."""
    results: dict[str, bool] = {}
    gen_script = BENCHMARK_PY / "generate_benchmark.py"

    for m in models:
        if m in TRAIN_MODELS and not train_results.get(m, True):
            print(f"\n  [SKIP] {m}: generation skipped (training failed)")
            results[m] = False
            continue

        print(f"\n{'='*60}")
        print(f"  GENERATING {m}")
        print(f"{'='*60}")
        ok = _run(gen_script, ["--model", m], f"generate {m}")
        results[m] = ok

    return results


# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(models: list[str],
                   gen_results: dict[str, bool]) -> bool:
    """Evaluate all models that have generated images."""
    eval_script = BENCHMARK_PY / "evaluate_benchmark.py"

    # Evaluate models that succeeded generation (or were already generated)
    models_to_eval = [m for m in models if gen_results.get(m, True)]
    if not models_to_eval:
        print("\n  [SKIP] Evaluation: no models have generated images")
        return False

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {', '.join(models_to_eval)}")
    print(f"{'='*60}")
    return _run(eval_script, ["--model", "all"], "evaluation")


# ─────────────────────────────────────────────────────────────────────────────
def print_summary(models: list[str], skip_train: bool,
                  train_res: dict, gen_res: dict, eval_ok: bool,
                  total_elapsed_min: float):
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")

    if not skip_train:
        print("\n  Training:")
        for m in models:
            if m not in TRAIN_MODELS:
                print(f"    {m:<4}: n/a (vanilla)")
            else:
                status = "OK" if train_res.get(m) else "FAILED"
                print(f"    {m:<4}: {status}")

    print("\n  Generation:")
    for m in models:
        status = "OK" if gen_res.get(m) else "FAILED/SKIPPED"
        print(f"    {m:<4}: {status}")

    print(f"\n  Evaluation: {'OK' if eval_ok else 'FAILED/SKIPPED'}")
    print(f"\n  Total elapsed: {total_elapsed_min:.1f} min")

    results_dir = REPO_ROOT / "benchmark" / "results"
    if results_dir.exists():
        print(f"\n  Results: {results_dir}")
        for f in sorted(results_dir.glob("*")):
            print(f"    {f.name}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run full DaiViet-Pattern benchmark pipeline"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training, go straight to generation + evaluation",
    )
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=(f"Comma-separated list of models to run "
              f"(default: {','.join(ALL_MODELS)})"),
    )
    args = parser.parse_args()

    models = [m.strip().upper() for m in args.models.split(",")]
    invalid = [m for m in models if m not in ALL_MODELS]
    if invalid:
        parser.error(f"Unknown model(s): {invalid}. "
                     f"Valid: {ALL_MODELS}")

    t_start = time.time()
    estimate_total_time(models, args.skip_train)

    train_results: dict[str, bool] = {}
    gen_results:   dict[str, bool] = {}
    eval_ok = False

    try:
        # ── Step 1: Train ──────────────────────────────────────────────────
        if not args.skip_train:
            train_results = run_training(models)
        else:
            print("\n  [INFO] --skip_train: skipping training step")
            train_results = {m: True for m in models}

        # ── Step 2: Generate ───────────────────────────────────────────────
        gen_results = run_generation(models, train_results)

        # ── Step 3: Evaluate ───────────────────────────────────────────────
        eval_ok = run_evaluation(models, gen_results)

    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Pipeline stopped by user.")

    total_elapsed_min = (time.time() - t_start) / 60
    print_summary(models, args.skip_train, train_results,
                  gen_results, eval_ok, total_elapsed_min)


if __name__ == "__main__":
    main()
