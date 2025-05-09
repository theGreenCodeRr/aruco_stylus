"""
Pose Prediction Analysis Script

This script loads annotated pose data and trains linear regression models to predict
input_angle from detected global translations (tglob_x, tglob_y, tglob_z) and from
local translations (tx_local, ty_local, tz_local). It plots predicted vs actual
input_angle for both models, includes the raw detected angle extracted from the
marker rotation vector and its calibrated counterpart, and creates comparison
plots vs frame.

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


def extract_detected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'detected_angle' from rglob_z (deg) and smooth with rolling median.
    """
    if {'rglob_x', 'rglob_y', 'rglob_z'}.issubset(df.columns):
        df['detected_angle'] = df['rglob_z'] * (180.0 / np.pi)
        df['detected_angle'] = df['detected_angle'].rolling(window=5, center=True, min_periods=1).median()
    else:
        print("Warning: rglob not found; skip detected.")
    return df


def analyze_global(df: pd.DataFrame, show_plots: bool = True) -> dict:
    """
    Predict input_angle from global translations, plot pred vs actual vs detected,
    and angle vs frame.
    """
    required = {'frame', 'tglob_x', 'tglob_y', 'tglob_z', 'input_angle'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns for global analysis: {missing}")

    X = df[['tglob_x', 'tglob_y', 'tglob_z']].values
    y = df['input_angle'].values.reshape(-1,1)
    frames = df['frame'].values
    has_detected = 'detected_angle' in df.columns

    # Train/test split
    if has_detected:
        detected = df['detected_angle'].values.reshape(-1,1)
        X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
            X, y, detected, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        d_test = None

    # Fit model
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results = {'mse': mse, 'mae': mae, 'r2': r2}

    # Calibrate detected
    if has_detected:
        calibrator = LinearRegression().fit(d_train, y_train)
        d_pred = calibrator.predict(d_test)
        results.update({
            'detected_mse': mean_squared_error(y_test, d_pred),
            'detected_mae': mean_absolute_error(y_test, d_pred),
            'detected_r2': r2_score(y_test, d_pred)
        })
    else:
        d_pred = None

    if show_plots:
        # Scatter plot
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, label='Predicted', s=40)
        if d_pred is not None:
            plt.scatter(y_test, d_pred, label='Detected', marker='x', s=40)
        mx = max(y_test.max(), y_pred.max(), (d_pred.max() if d_pred is not None else 0))
        plt.plot([0, mx], [0, mx], '--', label='Ideal')
        plt.xlabel('Actual Input Angle')
        plt.ylabel('Angle Value')
        plt.title('Global: Pred vs Actual vs Detected')
        plt.legend(); plt.tight_layout(); plt.show()

        # Angle vs Frame
        full_pred = model.predict(X).ravel()
        plt.figure(figsize=(8,4))
        plt.plot(frames, y.ravel(), '--', label='GT')
        plt.plot(frames, full_pred, '-', color='red', linewidth=2, label='Global Model')
        if d_pred is not None:
            full_det = df['detected_angle'].values
            full_cal = calibrator.predict(df['detected_angle'].values.reshape(-1,1)).ravel()
            plt.plot(frames, full_cal, ':', label='Detected(cal)')
        plt.xlabel('Frame'); plt.ylabel('Angle Value'); plt.title('Global Angle vs Frame')
        plt.legend(); plt.tight_layout(); plt.show()

    print(f"Global MSE={mse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
    if has_detected:
        print(f"Detected Cal MSE={results['detected_mse']:.3f}, MAE={results['detected_mae']:.3f}, R2={results['detected_r2']:.3f}")
    return results


def analyze_local(df: pd.DataFrame, show_plots: bool = True) -> dict:
    """
    Predict input_angle from local translations, plot pred vs actual vs detected,
    color-coded by marker_id, and angle vs frame comparison.
    """
    required = {'frame','tx_local','ty_local','tz_local','input_angle','marker_id'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns for local analysis: {missing}")

    X = df[['tx_local','ty_local','tz_local']].values
    y = df['input_angle'].values.reshape(-1,1)
    frames = df['frame'].values
    markers = df['marker_id'].astype(int).values
    has_detected = 'detected_angle' in df.columns

    if has_detected:
        detected = df['detected_angle'].values.reshape(-1,1)
        X_train, X_test, y_train, y_test, m_train, m_test, d_train, d_test = train_test_split(
            X, y, markers, detected, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            X, y, markers, test_size=0.2, random_state=42)
        d_test = None

    # Fit local model
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results = {'mse': mse, 'mae': mae, 'r2': r2}

    # Fit global model for overlay
    Xg = df[['tglob_x','tglob_y','tglob_z']].values
    global_model = LinearRegression().fit(Xg, y)
    global_pred = global_model.predict(Xg).ravel()

    if show_plots:
        # Scatter local pred vs actual vs detected
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, c=m_test, cmap=cmap, label='Predicted', s=40)
        if d_test is not None:
            plt.scatter(y_test, d_test, c=m_test, cmap=cmap, marker='x', label='Detected', s=40)
        mx = max(y_test.max(), y_pred.max(), (d_test.max() if d_test is not None else 0))
        plt.plot([0, mx], [0, mx], '--', label='Ideal')
        plt.xlabel('Actual Input Angle'); plt.ylabel('Angle Value')
        plt.title('Local: Pred vs Actual vs Detected')
        plt.colorbar(label='marker_id'); plt.legend(); plt.tight_layout(); plt.show()

        # Angle vs Frame per marker
        plt.figure(figsize=(8,4))
        unique_ids = np.unique(markers)
        for idx, mid in enumerate(unique_ids):
            mask = markers==mid
            fr = frames[mask]
            pred_full = model.predict(X[mask]).ravel()
            order = np.argsort(fr)
            plt.plot(fr[order], pred_full[order], label=f'Marker {mid}', color=cmap(idx))
        # Overlay global and GT
        plt.plot(frames, global_pred, '-', color='red', linewidth=2, label='Global Model')
        plt.plot(frames, y.ravel(), '--', color='black', label='GT')
        plt.xlabel('Frame'); plt.ylabel('Angle Value'); plt.title('Local & Global vs Frame')
        plt.legend(); plt.tight_layout(); plt.show()

        # Combined comparison
        plt.figure(figsize=(8,4))
        local_full = model.predict(X).ravel()
        plt.plot(frames, y.ravel(), '--', label='GT')
        plt.plot(frames, global_pred, '-.', color='red', linewidth=2, label='Global Model')
        plt.plot(frames, local_full, '-', color='blue', linewidth=2, label='Local Model')
        plt.xlabel('Frame'); plt.ylabel('Angle Value'); plt.title('Model Comparison vs Frame')
        plt.legend(); plt.tight_layout(); plt.show()

    print(f"Local MSE={mse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Pose Prediction Analysis')
    parser.add_argument('-c','--csv',default='annotated_poses.csv',help='Path to CSV')
    parser.add_argument('--no-plots',action='store_true',help='Disable plots')
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.csv): sys.exit(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    df = extract_detected(df)
    analyze_global(df,show_plots=not args.no_plots)
    analyze_local(df,show_plots=not args.no_plots)


# ===== Tests =====
def test_global_synth():
    n, x = 50, np.linspace(0,60,50)
    df = pd.DataFrame({'frame':np.arange(n),'tglob_x':x,'tglob_y':0,'tglob_z':0,'input_angle':3*x})
    m = analyze_global(df,show_plots=False)
    assert m['mse']<1e-8

def test_local_synth():
    n, x = 50, np.linspace(0,60,50)
    df = pd.DataFrame({
        'frame':np.arange(n),'tx_local':x,'ty_local':0,'tz_local':0,
        'input_angle':2*x,'marker_id':np.arange(n)%3,
        'tglob_x':x,'tglob_y':0,'tglob_z':0
    })
    m = analyze_local(df,show_plots=False)
    assert m['r2']>0.999

if __name__=='__main__':
    main(); print("Done.")
    test_global_synth(); print("Global test passed.")
    test_local_synth(); print("Local test passed.")
