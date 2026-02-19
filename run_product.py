import sys
import subprocess
import argparse

PY = sys.executable  # Always uses the active venv interpreter


def run_step(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command and return the CompletedProcess."""
    return subprocess.run(cmd, check=check)


def main() -> None:
    ap = argparse.ArgumentParser(description="RobustAI Engine - Product Demo Pipeline")
    ap.add_argument("--csv", default="data/factory_sensors.csv", help="Path to CSV dataset")
    ap.add_argument("--label", default="label", help="Label column name")
    ap.add_argument("--threshold", type=float, default=0.75, help="Safety threshold for effective accuracy")
    ap.add_argument("--model-name", default="MLP_v1", help="Model name for reporting/warnings")
    ap.add_argument("--skip-data-gen", action="store_true", help="Skip synthetic dataset generation step")
    args = ap.parse_args()

    print("--- 1) Generate Smart Factory Dataset ---")
    if args.skip_data_gen:
        print("Skipped dataset generation (--skip-data-gen).")
    else:
        run_step([PY, "-m", "src.data_gen"], check=True)

    print("\n--- 2) Run Week 06 Runner (Audit Mode) ---")
    p = run_step(
        [
            PY, "-m", "src.week06_runner",
            "--csv", args.csv,
            "--label", args.label,
            "--threshold", str(args.threshold),
            "--model-name", args.model_name,
        ],
        check=False,  # IMPORTANT: we want to continue even if it returns 2
    )
    runner_exit = int(p.returncode)

    # 0 = pass, 2 = safety violation (audit warning), other = real failure
    if runner_exit not in (0, 2):
        print(f"\n[ERROR] week06_runner exited with code {runner_exit}. Stopping before dashboard.")
        sys.exit(runner_exit)

    print("\n--- 3) Generate Dashboard ---")
    run_step([PY, "-m", "src.dashboard"], check=True)

    print("\nArtifacts:")
    print(" - results/week06_report.md")
    print(" - results/week06_product_demo_results.csv")
    print(" - results/week06_operating_envelope.csv")
    print(" - results/master_dashboard.png")

    if runner_exit == 2:
        print("\n[WARNING] Safety gate triggered (exit code 2). Artifacts were generated, but deployment is NOT recommended.")

    # Return the audit exit code so CI/deployment systems can block deployment automatically
    sys.exit(runner_exit)


if __name__ == "__main__":
    main()