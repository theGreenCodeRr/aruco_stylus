#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Script: angle_regression_pipelines.py
# Description: Performs angle regression on multiple
# global-pose pipelines and visualizes both
# regression scatter (styled) and time-series.
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Angle regression per pipeline')
    parser.add_argument('--gt', default='6DoF_annotated_poses.csv',
                        help='Ground-truth CSV (requires frame, rglob_x/y/z, input_angle, marker_id, tx_local/ty_local/tz_local)')
    parser.add_argument('--plain', default='plain_aruco_global.csv',
                        help='plain ArUco global CSV (frame, roll, pitch, yaw)')
    parser.add_argument('--qc', default='qc_global.csv',
                        help='QC-only global CSV (frame, roll, pitch, yaw)')
    parser.add_argument('--qc_kalman', default='qc_kalman_global.csv',
                        help='QC+Kalman global CSV (frame, roll, pitch, yaw)')
    parser.add_argument('--aruco_kalman', default='aruco_kalman_global.csv',
                        help='ArUco+Kalman global CSV (frame, roll, pitch, yaw)')
    parser.add_argument('--out_dir', default='results',
                        help='Output folder for results')
    return parser.parse_args()


def load_gt(gt_path):
    if not os.path.isfile(gt_path):
        print(f"Error: GT file '{gt_path}' not found.")
        sys.exit(1)
    df = pd.read_csv(gt_path)
    required = {'frame', 'rglob_x','rglob_y','rglob_z', 'input_angle', 'marker_id', 'tx_local','ty_local','tz_local'}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: GT CSV missing columns: {missing}")
        sys.exit(1)
    return df


def extract_euler_angles(df_gt):
    # Deduplicate by frame and convert Rodrigues -> Euler (degrees)
    df_u = df_gt.drop_duplicates('frame').reset_index(drop=True)
    rvecs = df_u[['rglob_x','rglob_y','rglob_z']].to_numpy(dtype=np.float32)
    rolls, pitches, yaws = [], [], []
    for r in rvecs:
        R, _ = cv2.Rodrigues(r)
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            rolls.append(np.degrees(np.arctan2(R[2,1], R[2,2])))
            pitches.append(np.degrees(np.arctan2(-R[2,0], sy)))
            yaws.append(np.degrees(np.arctan2(R[1,0], R[0,0])))
        else:
            rolls.append(np.degrees(np.arctan2(-R[1,2], R[1,1])))
            pitches.append(np.degrees(np.arctan2(-R[2,0], sy)))
            yaws.append(0.0)
    return pd.DataFrame({
        'frame': df_u['frame'],
        'input_angle': df_u['input_angle'],
        'roll_gt': rolls,
        'pitch_gt': pitches,
        'yaw_gt': yaws
    })


def styled_scatter_plot(df, method, out_path):
    # Sort by GT angle for smooth lines
    df_sorted = df.sort_values('input_angle')
    x = df_sorted['input_angle']
    y = df_sorted['pred_angle']

    # Compute metrics
    r2 = float(df['R2'].iloc[0]) if 'R2' in df.columns else None
    std = (df['pred_angle'] - df['input_angle']).std(ddof=0)

    plt.figure(figsize=(6,6))
    # GT line and Predicted line
    plt.plot(x, x, color='red',   linewidth=2, label='Ground Truth')
    plt.plot(x, y, color='green', linewidth=2, label='Predictions')

    # Labels & ticks
    plt.xlabel('Ground Truth', fontsize=16)
    plt.ylabel('Angle (deg)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Annotation bottom-right
    plt.text(
        0.95, 0.05,
        f'R² = {r2:.3f} (adaptive)',
        #f'R² = {r2:.3f} (QC)\nstandard deviation = {std:.2f}',
        transform=plt.gca().transAxes,
        ha='right', va='bottom', fontsize=16, fontweight='bold'
    )

    plt.legend(fontsize=16, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def time_series_plot(df, method, out_path):
    plt.figure(figsize=(8,4))
    plt.plot(df['frame'], df['input_angle'], 'r-', label='Ground Truth')
    plt.plot(df['frame'], df['pred_angle'], 'g-', label='Predictions')
    plt.xlabel('Frame', fontsize=16)
    plt.ylabel('Angle (deg)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_gt = load_gt(args.gt)
    df_angles = extract_euler_angles(df_gt)

    pipelines = {
        'plain_aruco':   args.plain,
        'qc':            args.qc,
        'qc_kalman':     args.qc_kalman,
        'aruco_kalman':  args.aruco_kalman
    }
    summary = []

    for method, path in pipelines.items():
        if not os.path.isfile(path):
            print(f"Warning: '{path}' not found, skipping {method}.")
            continue
        df_pred = pd.read_csv(path)
        if not {'frame','roll','pitch','yaw'}.issubset(df_pred.columns):
            print(f"Warning: '{method}' missing Euler columns, skipping.")
            continue

        # Merge and regression
        df_m = pd.merge(df_angles,
                        df_pred[['frame','roll','pitch','yaw']],
                        on='frame')
        df_m.rename(columns={'roll':'roll_pred','pitch':'pitch_pred','yaw':'yaw_pred'}, inplace=True)

        X = df_m[['roll_pred','pitch_pred','yaw_pred']].to_numpy()
        y = df_m['input_angle'].to_numpy()
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        r2 = model.score(X, y)

        df_m['pred_angle'] = preds
        df_m['error'] = preds - y

        summary.append({
            'method': method,
            'coef_roll':  model.coef_[0],
            'coef_pitch': model.coef_[1],
            'coef_yaw':   model.coef_[2],
            'intercept':  model.intercept_,
            'R2':         r2
        })

        # Save predictions
        df_m[['frame','input_angle','pred_angle','error']].to_csv(os.path.join(args.out_dir, f'predictions_{method}.csv'), index=False)

        # Styled scatter
        scatter_path = os.path.join(args.out_dir, f'angle_regression_{method}.png')
        styled_scatter_plot(df_m.assign(R2=r2), method, scatter_path)
        print(f"Saved styled scatter for {method} -> {scatter_path}")

        # Time-series
        ts_path = os.path.join(args.out_dir, f'time_series_{method}.png')
        time_series_plot(df_m, method, ts_path)
        print(f"Saved time-series for {method} -> {ts_path}")

    # Write summary CSV
    pd.DataFrame(summary).to_csv(
        os.path.join(args.out_dir, 'angle_regression_summary.csv'),
        index=False
    )
    print("Saved regression summary.")

if __name__ == '__main__':
    main()
