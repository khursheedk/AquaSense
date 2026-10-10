"""
generate_dataset.py — Synthetic dataset generator for AquaSense.

This script generates a clearly-labelled synthetic dataset for educational purposes.
The consumption model is based on documented per-activity water-use research:
  - Aquastat / FAO (https://www.fao.org/aquastat)
  - Ofwat household demand research (UK)
  - Australian Water Industry (AWA) per-capita benchmarks
  - US EPA WaterSense programme guidelines

The data is NOT real-world recorded data. It is a research-informed simulation
intended solely for a college machine-learning demonstration.
"""

import numpy as np
import pandas as pd

SEED = 42
N = 500  # number of synthetic households


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    occupants          = rng.integers(1, 9, size=n)                    # 1–8 people
    property_size_m2   = rng.uniform(40, 210, size=n).round(1)         # 40–210 m²
    num_bathrooms      = np.clip(rng.integers(1, 5, size=n), 1, 4)    # 1–4
    has_dishwasher     = rng.integers(0, 2, size=n)                    # 0 or 1
    has_washing_machine = rng.integers(0, 2, size=n)                   # 0 or 1
    avg_temp_celsius   = rng.uniform(10.0, 28.0, size=n).round(1)      # 10–28 °C
    garden_size_m2     = rng.uniform(0, 65, size=n).round(1)           # 0–65 m²
    days_away_per_year = rng.integers(0, 101, size=n)                  # 0–100 days

    # -----------------------------------------------------------------------
    # Research-backed consumption formula (litres per day):
    #   Base per-person use:   ~100 L/person/day  (Aquastat global average)
    #   Dishwasher cycle:      ~15 L/day when present
    #   Washing machine:       ~20 L/day when present
    #   Garden irrigation:     ~1.2 L/m²/day, scaled by temperature excess
    #   Absence correction:    proportional to days_away fraction
    #   Bathroom factor:       extra bathroom = slightly more use
    #   Noise:                 ±8 % Gaussian to reflect real variation
    # -----------------------------------------------------------------------
    temp_factor = np.clip((avg_temp_celsius - 15) / 13, 0, 1)  # 0 at 15 °C, 1 at 28 °C
    garden_use  = garden_size_m2 * 1.2 * temp_factor

    base = (
        occupants * 100
        + has_dishwasher * 15
        + has_washing_machine * 20
        + garden_use
        + (num_bathrooms - 1) * 8
    )

    absence_factor = 1 - (days_away_per_year / 365) * 0.4  # at most 40 % reduction
    base = base * absence_factor

    noise = rng.normal(loc=1.0, scale=0.08, size=n)
    consumption = np.clip(base * noise, 60, 700).round(1)

    df = pd.DataFrame({
        "occupants":            occupants,
        "property_size_m2":     property_size_m2,
        "num_bathrooms":        num_bathrooms,
        "has_dishwasher":       has_dishwasher,
        "has_washing_machine":  has_washing_machine,
        "avg_temp_celsius":     avg_temp_celsius,
        "garden_size_m2":       garden_size_m2,
        "days_away_per_year":   days_away_per_year,
        "water_consumption_lpd": consumption,   # litres per day — TARGET
    })

    return df


if __name__ == "__main__":
    df = generate()
    out = "water_consumption.csv"
    df.to_csv(out, index=False)
    print(f"Dataset saved -> {out}")
    print(f"Rows: {len(df)}  |  Columns: {list(df.columns)}")
    print(df.describe().round(2))
