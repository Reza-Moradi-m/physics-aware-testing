import subprocess
import sys

def main():
    py = sys.executable  # <-- uses your active venv python

    print("--- 1) Generate Smart Factory Dataset ---")
    subprocess.run([py, "-m", "src.data_gen"], check=True)

    print("\n--- 2) Run Week 06 Runner ---")
    subprocess.run([py, "-m", "src.week06_runner",
                    "--csv", "data/factory_sensors.csv",
                    "--label", "label"], check=True)

    print("\n--- 3) Generate Dashboard ---")
    subprocess.run([py, "-m", "src.dashboard"], check=True)

if __name__ == "__main__":
    main()