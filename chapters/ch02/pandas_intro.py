"""Introduction to Pandas for EEG data analysis.

Demonstrates creating a DataFrame of EEG features, summary statistics,
filtering, and adding computed columns.

Usage:
    python pandas_intro.py
"""

import numpy as np
import pandas as pd


def main() -> None:
    # Create a DataFrame of EEG features from 4 channels
    # This simulates features extracted for ML input
    np.random.seed(42)
    channels = ['P4', 'Cz', 'F8', 'T7']
    data = {
        'channel': channels,
        'mean_uV': np.random.randn(4) * 5,
        'std_uV': np.random.rand(4) * 40 + 30,
        'alpha_power': np.random.rand(4) * 10,
        'theta_power': np.random.rand(4) * 5,
    }

    df = pd.DataFrame(data)
    print("EEG features table:")
    print(df)
    print()

    # Summary statistics
    print("Summary statistics:")
    print(df.describe())
    print()

    # Select channels with high alpha power
    high_alpha = df[df['alpha_power'] > 5.0]
    print(f"Channels with alpha_power > 5.0:")
    print(high_alpha[['channel', 'alpha_power']])
    print()

    # Add a new column (alpha/theta ratio)
    df['alpha_theta_ratio'] = df['alpha_power'] / df['theta_power']
    print("With alpha/theta ratio:")
    print(df[['channel', 'alpha_theta_ratio']])


if __name__ == "__main__":
    main()
