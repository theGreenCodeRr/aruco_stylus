"""
Pose Prediction Analysis Script

This script loads annotated pose data and trains a linear regression model to predict
input_angle from detected global translations (tglob_x, tglob_y, tglob_z). It then plots
predicted vs actual input_angle, alongside the raw detected angle extracted from the
marker rotation vector and its calibrated counterpart, using the 'frame' column for x-axis.

Usage:
    python pose_analysis.py --csv /path/to/annotated_poses.csv

Requirements:
    pip install pandas numpy scikit-learn matplotlib
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def analyze_global(df: pd.DataFrame, show_plots: bool = True) -> dict:
    """
    Train a LinearRegression to predict input_angle from global translations.
    Optionally include raw detected angle (from rotation vector), calibrate it,
    and plot actual vs predicted vs detected_calibrated input_angle using 'frame' values.

    Returns:
        metrics: dict with keys 'mse', 'mae', 'r2', 'detected_mse', 'detected_mae', 'detected_r2'
    """
    # Verify required columns
    required = {'frame', 'tglob_x', 'tglob_y', 'tglob_z', 'input_angle'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    # Features and target
    X = df[['tglob_x', 'tglob_y', 'tglob_z']].values
    y = df['input_angle'].values.reshape(-1, 1)
    frames = df['frame'].values

    # Detect and calibrate if available
    has_detected = 'detected_angle' in df.columns
    if has_detected:
        detected = df['detected_angle'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
            X, y, detected, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        d_train = d_test = None

    # Train and predict
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {'mse': mse, 'mae': mae, 'r2': r2}

    # Calibrate detected, if present
    if has_detected:
        calibrator = LinearRegression().fit(d_train, y_train)
        d_pred = calibrator.predict(d_test)
        dmse = mean_squared_error(y_test, d_pred)
        dmae = mean_absolute_error(y_test, d_pred)
        dr2 = r2_score(y_test, d_pred)
        metrics.update({'detected_mse': dmse, 'detected_mae': dmae, 'detected_r2': dr2})
    else:
        d_pred = None

    # Plot results
    if show_plots:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test.ravel(), y_pred.ravel(), label='Predicted', s=40)
        if has_detected:
            plt.scatter(y_test.ravel(), d_pred.ravel(), label='Detected (calibrated)', marker='x', s=40)
        max_val = max(y_test.max(), y_pred.max(), d_pred.max() if d_pred is not None else y_pred.max())
        plt.plot([0, max_val], [0, max_val], '--', label='Ideal')
        plt.xlabel('Actual Input Angle')
        plt.ylabel('Angle Value')
        plt.title('Global: Predicted vs Actual vs Detected (calibrated)')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(frames, y.ravel(), label='GT')
        full_pred = model.predict(X)
        plt.plot(frames, full_pred.ravel(), label='Predicted')
        if has_detected:
            full_detected = df['detected_angle'].values.reshape(-1, 1)
            full_cal = calibrator.predict(full_detected)
            plt.plot(frames, full_cal.ravel(), label='Detected (calibrated)', linestyle=':')
        plt.xlabel('Frame')
        plt.ylabel('Angle Value')
        plt.title('Angle Comparison Across Frames')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Print metrics
    print(f"Global MSE={mse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
    if has_detected:
        print(f"Detected Calibrated MSE={dmse:.3f}, MAE={dmae:.3f}, R2={dr2:.3f}")
    return metrics


def extract_detected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'detected_angle' column extracted from the rotation vector columns
    (rglob_x, rglob_y, rglob_z). Assumes rotation about Z-axis,
    so uses the Z component (rglob_z) converted from radians to degrees and
    applies a rolling median filter to smooth noise.
    """
    if {'rglob_x', 'rglob_y', 'rglob_z'}.issubset(df.columns):
        df['detected_angle'] = df['rglob_z'] * (180.0 / np.pi)
        df['detected_angle'] = df['detected_angle'].rolling(window=5, center=True, min_periods=1).median()
    else:
        print("Warning: Rotation vector columns (rglob_*) not found. 'detected_angle' skipped.")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Global Angle Prediction Analysis')
    parser.add_argument('-c', '--csv', default='annotated_poses.csv', help='Path to annotated_poses.csv')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.csv):
        sys.exit(f"Error: CSV not found at '{args.csv}'")
    df = pd.read_csv(args.csv)
    df = extract_detected(df)
    analyze_global(df, show_plots=not args.no_plots)


# ===== Tests =====

def test_analyze_global_synthetic():
    n = 50
    x = np.linspace(0, 60, n)
    df = pd.DataFrame({
        'frame': np.arange(n), 'tglob_x': x, 'tglob_y': np.zeros(n), 'tglob_z': np.zeros(n),
        'input_angle': 3 * x
    })
    metrics = analyze_global(df, show_plots=False)
    assert metrics['mse'] < 1e-8, f"MSE should be near zero, got {metrics['mse']}"


def test_analyze_with_detected_noise():
    n = 50
    x = np.linspace(0, 60, n)
    df = pd.DataFrame({
        'frame': np.arange(n), 'tglob_x': x, 'tglob_y': np.zeros(n), 'tglob_z': np.zeros(n),
        'input_angle': 3 * x,
        'rglob_x': np.zeros(n), 'rglob_y': np.zeros(n), 'rglob_z': np.zeros(n)
    })
    df['detected_angle'] = 3 * x + np.random.randn(n) * 0.1
    metrics = analyze_global(df, show_plots=False)
    assert 'detected_mse' in metrics, "Calibration metrics missing for detected"


if __name__ == '__main__':
    main()
    print("Analysis completed.")
    test_analyze_global_synthetic()
    print("test_analyze_global_synthetic passed.")
    test_analyze_with_detected_noise()
    print("test_analyze_with_detected_noise passed.")
