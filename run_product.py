# run_product.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PY = sys.executable


def run_step(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check)


def ensure_nasa_dataset(csv_path: str) -> None:
    csv_file = Path(csv_path)
    if csv_file.exists():
        print(f"NASA dataset already exists: {csv_file}")
        return

    print("Setting up NASA dataset...")
    run_step([PY, "-m", "src.adapters.nasa_adapter"], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="RobustAI Engine - Product Pipeline")
    ap.add_argument("--dataset", default="nasa", choices=["nasa"], help="Dataset adapter to use")
    ap.add_argument("--csv", default="data/nasa_jet_engine.csv", help="Path to standardized CSV")
    ap.add_argument("--label", default="label", help="Label column name")
    ap.add_argument("--profile", default="aviation_extreme", help="Deployment profile")
    ap.add_argument("--dataset-name", default="nasa_fd001", help="Dataset name for reports")
    ap.add_argument("--compare", action="store_true", help="Run model comparison")
    ap.add_argument("--save-gold", action="store_true", help="Save run as regression gold baseline")
    ap.add_argument("--pdf", action="store_true", help="Generate executive PDF if your existing PDF module supports current outputs")
    args = ap.parse_args()

    print("--- ROBUSTAI ENGINE: PRODUCT AUDIT ---")

    if args.dataset == "nasa":
        ensure_nasa_dataset(args.csv)

    print("\n--- RUNNING AUDIT ---")
    cmd = [
        PY,
        "-m",
        "src.week06_runner",
        "--csv",
        args.csv,
        "--label",
        args.label,
        "--profile",
        args.profile,
        "--dataset-name",
        args.dataset_name,
    ]

    if args.compare:
        cmd.append("--compare")
    if args.save_gold:
        cmd.append("--save-gold")

    proc = run_step(cmd, check=False)
    runner_exit = int(proc.returncode)

    if runner_exit not in (0, 2):
        print(f"[ERROR] Runner failed with exit code {runner_exit}")
        raise SystemExit(runner_exit)

    print("\n--- GENERATING DASHBOARD ---")
    dashboard_cmd = [PY, "-m", "src.dashboard"]
    dash_proc = run_step(dashboard_cmd, check=False)
    if dash_proc.returncode != 0:
        print("[WARNING] Dashboard generation failed. Check src/dashboard.py against new results format.")

    if args.pdf:
        print("\n--- GENERATING EXECUTIVE PDF ---")
        pdf_cmd = [PY, "-m", "src.pdf_report"]
        pdf_proc = run_step(pdf_cmd, check=False)
        if pdf_proc.returncode != 0:
            print("[WARNING] PDF generation failed. Check src/pdf_report.py against new results format.")

    print("\nArtifacts:")
    print(" - results/week06_report.md")
    print(" - results/week06_product_demo_results.csv")
    print(" - results/feature_sensitivity.csv")
    print(" - results/week06_operating_envelope.csv")
    print(" - results/mitigations.md")
    print(" - results/audit.log")
    print(" - results/regression_report.json (if baseline exists)")
    print(" - results/regression_summary.md (if baseline exists)")
    print(" - results/gold_standard.json (if saved)")
    print(" - results/master_dashboard.png (if dashboard succeeds)")
    if args.pdf:
        print(" - results/executive_report.pdf (if PDF succeeds)")

    try:
        results = pd.read_csv("results/week06_product_demo_results.csv")
        worst_acc = results["effective_accuracy"].min()
        print(f"\nWorst effective accuracy observed: {worst_acc:.4f}")
    except Exception:
        pass

    if runner_exit == 2:
        print("\n[WARNING] Safety gate triggered. Artifacts were generated, but deployment is NOT recommended.")

    raise SystemExit(runner_exit)


if __name__ == "__main__":
    main()