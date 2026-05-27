import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ROOT = (
    PROJECT_ROOT
    / "voltage_prediction_and_ISC_detection-V1.0"
    / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
)

ISC_ROOT = DATASET_ROOT / "NCM811_ISC_TEST"
OUTPUT_ROOT = DATASET_ROOT / "Noise"

TARGET_MODES = ("CC", "DST")
NOISE_STDS = (0.01, 0.05, 0.1)
RANDOM_SEED = 42

# load_right_block() uses the right-side voltage column from the raw CSV.
RIGHT_BLOCK_VOLTAGE_COLUMN_INDEX = 10


def add_voltage_noise(input_path, output_path, noise_std, rng):
    df = pd.read_csv(input_path)

    if len(df.columns) <= RIGHT_BLOCK_VOLTAGE_COLUMN_INDEX:
        raise ValueError(f"Voltage column is missing: {input_path}")

    voltage = pd.to_numeric(
        df.iloc[:, RIGHT_BLOCK_VOLTAGE_COLUMN_INDEX],
        errors="coerce",
    )
    noise = rng.normal(0.0, noise_std, size=len(df))

    noisy_voltage = voltage.copy()
    valid_voltage = voltage.notna()
    noisy_voltage.loc[valid_voltage] = voltage.loc[valid_voltage] + noise[valid_voltage]

    df.iloc[:, RIGHT_BLOCK_VOLTAGE_COLUMN_INDEX] = noisy_voltage

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_noisy_isc_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    saved_count = 0

    for mode in TARGET_MODES:
        source_dir = ISC_ROOT / mode
        csv_paths = sorted(source_dir.glob("*.csv"))

        if not csv_paths:
            print(f"Skip {mode}: no CSV files found in {source_dir}")
            continue

        for noise_std in NOISE_STDS:
            noise_dir_name = f"noise_{noise_std:g}"
            output_dir = OUTPUT_ROOT / mode / noise_dir_name

            for csv_path in csv_paths:
                output_path = output_dir / csv_path.name
                add_voltage_noise(csv_path, output_path, noise_std, rng)
                saved_count += 1

            print(f"Saved {len(csv_paths)} files: {output_dir}")

    print(f"Done. Saved total {saved_count} noisy CSV files.")


def main():
    build_noisy_isc_dataset()


if __name__ == "__main__":
    main()
