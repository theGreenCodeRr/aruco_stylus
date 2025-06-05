#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
Extract global Euler angles from 6DoF Rodrigues vectors and fit a multiple regression
model to predict the ground-truth scalar `input_angle` from (roll, pitch, yaw),
compute error statistics, and visualize predicted vs. ground truth.

Usage:
  python angle_regression.py --data 6DoF_annotated_poses.csv --out predictions.csv

Outputs:
  - Prints learned equation, R², prediction error mean and SD
  - Writes CSV with [frame, input_angle, pred_angle, error]
  - Saves time-series plot `pred_vs_gt.png`
"""

def load_6dof(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        print(f"Error: file '{csv_path}' not found.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    required = {'frame', 'rglob_x', 'rglob_y', 'rglob_z', 'input_angle'}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: missing columns {missing}")
        sys.exit(1)
    return df


def extract_euler(df: pd.DataFrame) -> pd.DataFrame:
    # Keep one entry per frame
    df_u = df.drop_duplicates('frame').reset_index(drop=True)
    rvecs = df_u[['rglob_x','rglob_y','rglob_z']].to_numpy(dtype=np.float32)
    rolls, pitches, yaws = [], [], []
    for r in rvecs:
        R, _ = cv2.Rodrigues(r)
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
            pitch = np.degrees(np.arctan2(-R[2,0], sy))
            yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
        else:
            roll  = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            pitch = np.degrees(np.arctan2(-R[2,0], sy))
            yaw   = 0.0
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)
    return pd.DataFrame({
        'frame': df_u['frame'],
        'input_angle': df_u['input_angle'],
        'roll': rolls,
        'pitch': pitches,
        'yaw': yaws,
    })


def run_regression(df: pd.DataFrame) -> pd.DataFrame:
    X = df[['roll','pitch','yaw']].to_numpy()
    y = df['input_angle'].to_numpy()
    model = LinearRegression().fit(X, y)
    coef = model.coef_
    intercept = model.intercept_
    r2 = model.score(X, y)
    print(f"GT_angle = {coef[0]:.3f}*roll + {coef[1]:.3f}*pitch + {coef[2]:.3f}*yaw + {intercept:.3f}")
    print(f"R² = {r2:.3f}")
    df['pred_angle'] = model.predict(X)
    df['error'] = df['pred_angle'] - df['input_angle']
    mean_err = df['error'].mean()
    std_err = df['error'].std()
    print(f"Prediction error: mean = {mean_err:.3f} deg, std = {std_err:.3f} deg")
    return df


def visualize(df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8,4))
    plt.plot(df['frame'], df['input_angle'], 'r-', label='GT')
    plt.plot(df['frame'], df['pred_angle'], 'g-', label='Pred')
    plt.xlabel('Frame')
    plt.ylabel('Angle (deg)')
    plt.title('Ground Truth vs. Predicted Angle over Frames')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Angle regression with error stats and visualization')
    parser.add_argument('--data','-d', default='6DoF_annotated_poses.csv', help='6DoF CSV path')
    parser.add_argument('--out','-o', default='predictions.csv', help='Output CSV with predictions')
    args = parser.parse_args()

    df6 = load_6dof(args.data)
    df_angles = extract_euler(df6)
    df_pred = run_regression(df_angles)

    # Save predictions and error
    df_pred[['frame','input_angle','pred_angle','error']].to_csv(args.out, index=False)
    print(f"Saved predictions and error to {args.out}")

    # Visualize time-series
    visualize(df_pred, 'pred_vs_gt.png')
    print("Saved plot to pred_vs_gt.png")

if __name__ == '__main__':
    main()
