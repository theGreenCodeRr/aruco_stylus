# -*- coding: utf-8 -*-
import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def extract_detected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 'detected_angle' from rotation vectors (rglob_z), smoothing via Savitzky-Golay.
    """
    if {'rglob_x', 'rglob_y', 'rglob_z'}.issubset(df.columns):
        raw = df['rglob_z'] * (180.0 / np.pi)
        window = min(len(raw) if len(raw) % 2 == 1 else len(raw) - 1, 21)
        window = max(window, 5)
        df['detected_angle'] = savgol_filter(raw, window_length=window, polyorder=2)
        # further smooth with rolling median
        df['detected_angle'] = df['detected_angle'].rolling(window=11, center=True, min_periods=1).median()
    else:
        print("Warning: Rotation vector columns not found. Skipping 'detected_angle'.")
    return df


def analyze_global(df: pd.DataFrame):
    X = df[['tglob_x', 'tglob_y', 'tglob_z']].values
    y = df['input_angle'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, {
        'global_mse': mean_squared_error(y_test, y_pred),
        'global_mae': mean_absolute_error(y_test, y_pred),
        'global_r2': r2_score(y_test, y_pred)
    }


def analyze_local_per_marker(df: pd.DataFrame):
    local_models = {}
    local_metrics = {}
    for mid in df['marker_id'].unique():
        sub = df[df['marker_id'] == mid].copy()
        if {'tx_local', 'ty_local', 'tz_local'}.issubset(sub.columns):
            feats = ['tx_local', 'ty_local', 'tz_local']
        elif {'rloc_x', 'rloc_y', 'rloc_z'}.issubset(sub.columns):
            feats = ['rloc_x', 'rloc_y', 'rloc_z']
        else:
            continue
        if len(sub) < 5:
            continue
        X = sub[feats].values
        y = sub['input_angle'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        local_models[mid] = (model, feats)
        local_metrics[f'local_{mid}_mse'] = mean_squared_error(y_test, y_pred)
        local_metrics[f'local_{mid}_mae'] = mean_absolute_error(y_test, y_pred)
        local_metrics[f'local_{mid}_r2'] = r2_score(y_test, y_pred)
    return local_models, local_metrics


def analyze_detected(df: pd.DataFrame):
    """
    Calibrate detected_angle -> input_angle using polynomial regression.
    """
    d = df['detected_angle'].values.reshape(-1, 1)
    y = df['input_angle'].values.reshape(-1, 1)
    d_train, d_test, y_train, y_test = train_test_split(d, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lr', LinearRegression())
    ])
    pipeline.fit(d_train, y_train)
    y_pred = pipeline.predict(d_test)
    return pipeline, {
        'detected_mse': mean_squared_error(y_test, y_pred),
        'detected_mae': mean_absolute_error(y_test, y_pred),
        'detected_r2': r2_score(y_test, y_pred)
    }


def analyze_and_plot(df: pd.DataFrame, show_plots: bool = True) -> dict:
    df = extract_detected(df)
    g_model, g_metrics = analyze_global(df)
    local_models, l_metrics = analyze_local_per_marker(df) if 'marker_id' in df.columns else ({}, {})
    det_model, d_metrics = analyze_detected(df) if 'detected_angle' in df.columns else (None, {})
    metrics = {**g_metrics, **l_metrics, **d_metrics}

    if show_plots:
        frames = df['frame'].values
        gt = df['input_angle'].values
        plt.figure(figsize=(12, 6))
        plt.plot(frames, gt, '--', label='GT', linewidth=2)
        y_global = g_model.predict(df[['tglob_x', 'tglob_y', 'tglob_z']].values).ravel()
        plt.plot(frames, y_global, label='Global', linewidth=4)
        for mid, (model, feats) in local_models.items():
            sub = df[df['marker_id'] == mid].sort_values('frame')
            y_local = model.predict(sub[feats].values).ravel()
            plt.plot(sub['frame'], y_local, label=f'Local {mid}', linewidth=2)
        if det_model:
            y_det = det_model.predict(df['detected_angle'].values.reshape(-1, 1)).ravel()
            plt.plot(frames, y_det, label='Detected Poly', linewidth=2)
        plt.xlabel('Frame')
        plt.ylabel('Angle (deg)')
        plt.title('Predictions vs GT')
        plt.legend()
        plt.tight_layout()
        plt.show()

    for k, v in metrics.items():
        print(f"{k} = {v:.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Pose Prediction Analysis')
    parser.add_argument('-c', '--csv', default='CSVs/6DoF_annotated_poses.csv', help='CSVs/6DoF_annotated_poses.csv')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    args = parser.parse_args()
    if not os.path.isfile(args.csv):
        sys.exit(f"Error: CSV not found at '{args.csv}'")
    df = pd.read_csv(args.csv)
    analyze_and_plot(df, show_plots=not args.no_plots)

if __name__ == '__main__':
    main()
