"""
data_generation.py
Generates a synthetic dataset simulating programmer work patterns.
~2000 samples with 10 features + target label (productivity_level).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
N = 2000


def generate_dataset(n=N, save_path="dataset.csv"):
    """Generate synthetic programmer productivity dataset."""

    # --- Base latent productivity factor (hidden) ---
    latent = np.random.normal(0, 1, n)

    # Features correlated with latent productivity
    coding_hours        = np.clip(4 + 2 * latent + np.random.normal(0, 1.2, n), 0, 12)
    commits_per_day     = np.clip(2 + 1.5 * latent + np.random.normal(0, 1, n), 0, 15)
    lines_of_code       = np.clip(150 + 80 * latent + np.random.normal(0, 40, n), 10, 600)
    bugs_fixed          = np.clip(1 + 0.8 * latent + np.random.normal(0, 0.8, n), 0, 10)
    sleep_hours         = np.clip(7 + 0.5 * latent + np.random.normal(0, 0.8, n), 4, 10)
    distractions        = np.clip(5 - 1.5 * latent + np.random.normal(0, 1.5, n), 0, 15)
    task_completion_rate= np.clip(0.5 + 0.2 * latent + np.random.normal(0, 0.1, n), 0.0, 1.0)
    meetings_per_day    = np.clip(2 - 0.3 * latent + np.random.normal(0, 1, n), 0, 8)
    break_time          = np.clip(30 + 5 * latent + np.random.normal(0, 10, n), 0, 90)
    focus_score         = np.clip(50 + 15 * latent + np.random.normal(0, 10, n), 0, 100)

    # Compute productivity score (weighted sum)
    score = (
        0.20 * (coding_hours / 12) * 100 +
        0.15 * (commits_per_day / 15) * 100 +
        0.15 * (lines_of_code / 600) * 100 +
        0.10 * (bugs_fixed / 10) * 100 +
        0.10 * (sleep_hours / 10) * 100 +
        0.10 * (1 - distractions / 15) * 100 +
        0.10 * task_completion_rate * 100 +
        0.05 * (1 - meetings_per_day / 8) * 100 +
        0.05 * (focus_score / 100) * 100
    )
    score = np.clip(score, 0, 100)

    # Map score → productivity_level
    def label(s):
        if s < 40:
            return "Low"
        elif s < 70:
            return "Medium"
        else:
            return "High"

    productivity_level = [label(s) for s in score]

    df = pd.DataFrame({
        "coding_hours":         np.round(coding_hours, 2),
        "commits_per_day":      np.round(commits_per_day, 2),
        "lines_of_code":        np.round(lines_of_code, 0).astype(int),
        "bugs_fixed":           np.round(bugs_fixed, 2),
        "sleep_hours":          np.round(sleep_hours, 2),
        "distractions":         np.round(distractions, 2),
        "task_completion_rate": np.round(task_completion_rate, 4),
        "meetings_per_day":     np.round(meetings_per_day, 2),
        "break_time":           np.round(break_time, 2),
        "focus_score":          np.round(focus_score, 2),
        "productivity_level":   productivity_level,
    })

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[✓] Dataset saved to '{save_path}' — {len(df)} rows")
        print(df["productivity_level"].value_counts())

    return df


if __name__ == "__main__":
    generate_dataset(save_path="dataset.csv")
