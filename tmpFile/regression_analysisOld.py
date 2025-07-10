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
Perform angle regression for each pipeline's global Euler estimates and visualize time-series.
Assumes CSVs in current directory:
  - 6DoF_annotated_poses.csv
  - plain_aruco_global.csv
  - qc_global.csv
  - qc_kalman_global.csv
  - aruco_kalman_global.csv
Outputs:
  - angle_regression_summary.csv (one row per pipeline: coefficients + R²)
  - predictions_{method}.csv, angle_regression_{method}.png (scatter), and time_series_{method}.png
    for each pipeline
  - per_marker_local_translation_regression.csv and local_translation_norm_scatter.png
  - per_marker_local_angle_regression.csv and per_marker_{ID}_angle_timeseries.png (for each marker)

Usage:
  python angle_regression_pipelines.py

Optional flags to override filenames:
  --gt GT.csv --plain PLAIN.csv --qc QC.csv --qc_kalman QC_KALMAN.csv \
  --aruco_kalman ARUCO_KALMAN.csv --out_dir results_folder
"""

# ----------------------------
# Arguments and file paths (with defaults)
# ----------------------------
parser = argparse.ArgumentParser(description='Angle regression per pipeline')
parser.add_argument('--gt', default='6DoF_annotated_poses.csv',
                    help='Ground-truth CSV (must contain: frame, rglob_x/y/z, input_angle, tglob_x/y/z, marker_id, tx_local/ty_local/tz_local)')
parser.add_argument('--plain', default='plain_aruco_global.csv',
                    help='plain ArUco global CSV (must contain: frame, roll, pitch, yaw)')
parser.add_argument('--qc', default='qc_global.csv',
                    help='QC-only global CSV (must contain: frame, roll, pitch, yaw)')
parser.add_argument('--qc_kalman', default='qc_kalman_global.csv',
                    help='QC+Kalman global CSV (must contain: frame, roll, pitch, yaw)')
parser.add_argument('--aruco_kalman', default='aruco_kalman_global.csv',
                    help='ArUco+Kalman global CSV (must contain: frame, roll, pitch, yaw)')
parser.add_argument('--out_dir', default='results',
                    help='Output folder for results')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ----------------------------
# Load ground truth
# ----------------------------
def load_gt(gt_path):
    if not os.path.isfile(gt_path):
        print(f"Error: GT file '{gt_path}' not found.")
        sys.exit(1)
    df = pd.read_csv(gt_path)
    req = {
        'frame',
        'rglob_x', 'rglob_y', 'rglob_z',
        'input_angle',
        'tglob_x', 'tglob_y', 'tglob_z',
        'marker_id',
        'tx_local', 'ty_local', 'tz_local'
    }
    missing = req - set(df.columns)
    if missing:
        print(f"Error: GT missing columns {missing}")
        sys.exit(1)
    return df

df_gt = load_gt(args.gt)

# ----------------------------
# Extract Euler angles from rotation vectors (GT)
# ------------------------------------------
def extract_euler(df):
    df_u = df.drop_duplicates('frame').reset_index(drop=True)
    rvecs = df_u[['rglob_x', 'rglob_y', 'rglob_z']].to_numpy(dtype=np.float32)
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
        'roll_gt': rolls,
        'pitch_gt': pitches,
        'yaw_gt': yaws
    })

df_angles = extract_euler(df_gt)

# ----------------------------
# Global angle regression and visualization
# ----------------------------
pipelines = {
    'plain_aruco':   args.plain,
    'qc':            args.qc,
    'qc_kalman':     args.qc_kalman,
    'aruco_kalman':  args.aruco_kalman
}
summary_rows = []

for method, csv_path in pipelines.items():
    if not os.path.isfile(csv_path):
        print(f"Warning: File '{csv_path}' not found. Skipping {method}.")
        continue
    df_pred = pd.read_csv(csv_path)
    if not {'frame', 'roll', 'pitch', 'yaw'}.issubset(df_pred.columns):
        print(f"Warning: '{method}' missing Euler columns. Skipping.")
        continue

    # Merge GT‐angles with predicted‐angles on frame
    df_merge = pd.merge(
        df_angles,
        df_pred[['frame', 'roll', 'pitch', 'yaw']],
        on='frame'
    ).rename(columns={'roll':'roll_pred', 'pitch':'pitch_pred', 'yaw':'yaw_pred'})

    # 1) Fit linear model: input_angle ~ (roll_pred, pitch_pred, yaw_pred)
    X = df_merge[['roll_pred', 'pitch_pred', 'yaw_pred']].to_numpy()
    y = df_merge['input_angle'].to_numpy()
    model = LinearRegression().fit(X, y)
    coef = model.coef_
    intercept = model.intercept_
    r2 = model.score(X, y)

    # Record summary
    summary_rows.append({
        'method':     method,
        'coef_roll':  coef[0],
        'coef_pitch': coef[1],
        'coef_yaw':   coef[2],
        'intercept':  intercept,
        'R2':         r2
    })

    # 2) Compute predictions + errors
    df_merge['pred_angle'] = model.predict(X)
    df_merge['error'] = df_merge['pred_angle'] - df_merge['input_angle']
    err_mean = df_merge['error'].mean()
    err_std  = df_merge['error'].std()

    print(f"[{method}] angle = {coef[0]:.3f}*roll + {coef[1]:.3f}*pitch + {coef[2]:.3f}*yaw + {intercept:.3f}")
    print(f"[{method}]    R² = {r2:.3f}, err_mean = {err_mean:.3f}, err_std = {err_std:.3f}")

    # 3) Save per-frame predictions
    out_csv = os.path.join(args.out_dir, f"predictions_{method}.csv")
    df_merge[['frame','input_angle','pred_angle','error']].to_csv(out_csv, index=False)

    # 4a) Scatter plot: GT angle vs Predicted angle (with metrics)
    plt.figure(figsize=(6,6))
    plt.scatter(df_merge['input_angle'], df_merge['pred_angle'], s=10, alpha=0.6)
    mn = min(df_merge['input_angle'].min(), df_merge['pred_angle'].min())
    mx = max(df_merge['input_angle'].max(), df_merge['pred_angle'].max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.xlabel('GT input_angle')
    plt.ylabel('Predicted input_angle')
    plt.title(f"{method}: Angle Regression  (R²={r2:.3f}, μ_err={err_mean:.3f}, σ_err={err_std:.3f})")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"angle_regression_{method}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved angle regression scatter for '{method}' to '{fig_path}'")

    # 4b) Time-series: frame vs GT / Predicted angle
    plt.figure(figsize=(8,4))
    plt.plot(df_merge['frame'], df_merge['input_angle'], 'r-', label='GT')
    plt.plot(df_merge['frame'], df_merge['pred_angle'],  'g-', label='Pred')
    plt.xlabel('Frame')
    plt.ylabel('Angle (deg)')
    plt.title(f"{method}: GT vs Pred Angle over Frames")
    plt.legend()
    plt.tight_layout()
    ts_path = os.path.join(args.out_dir, f"time_series_{method}.png")
    plt.savefig(ts_path)
    plt.close()
    print(f"Saved time-series angle plot for '{method}' to '{ts_path}'")

# Save the global regression summary
pd.DataFrame(summary_rows).to_csv(
    os.path.join(args.out_dir, 'angle_regression_summary.csv'),
    index=False
)
print("Saved global angle regression summary to 'angle_regression_summary.csv'")

# ----------------------------
# Per-Marker Local Translation + Angle Regression
# ------------------------------------------
# Load predicted local data from `plain_aruco_local.csv` (frame, marker_id, tx, ty, tz, roll, pitch, yaw)
pred_local_file = '../plain_aruco_local.csv'

if not os.path.isfile(pred_local_file):
    print(f"Warning: '{pred_local_file}' not found. Skipping local regression.")
else:
    df_pred_local = pd.read_csv(pred_local_file)
    # Ensure necessary columns exist
    if not {'frame','marker_id','tx','ty','tz','roll','pitch','yaw'}.issubset(df_pred_local.columns):
        print(f"Warning: '{pred_local_file}' missing required local columns. Skipping.")
    else:
        # --- Translation Regression ---
        gt_local_trans = df_gt[['frame','marker_id','tx_local','ty_local','tz_local']]
        df_tr = pd.merge(
            gt_local_trans,
            df_pred_local[['frame','marker_id','tx','ty','tz']].rename(
                columns={'tx':'tx_pred','ty':'ty_pred','tz':'tz_pred'}
            ),
            on=['frame','marker_id']
        )
        trans_rows = []
        for mid in sorted(df_tr['marker_id'].unique()):
            df_m = df_tr[df_tr['marker_id'] == mid].copy()
            X_t = df_m[['tx_pred','ty_pred','tz_pred']].to_numpy()
            Y_t = df_m[['tx_local','ty_local','tz_local']].to_numpy()
            X_aug_t = np.hstack([np.ones((len(X_t),1)), X_t])
            Beta_t, _, _, _ = np.linalg.lstsq(X_aug_t, Y_t, rcond=None)
            Y_pred_t = X_aug_t.dot(Beta_t)
            ss_res_t = np.sum((Y_t - Y_pred_t)**2, axis=0)
            ss_tot_t = np.sum((Y_t - np.mean(Y_t, axis=0))**2, axis=0)
            r2_t = 1 - (ss_res_t / ss_tot_t)
            trans_rows.append({
                'marker_id': mid,
                'int_x': Beta_t[0,0], 'slope_tx_x': Beta_t[1,0], 'slope_ty_x': Beta_t[2,0], 'slope_tz_x': Beta_t[3,0], 'R2_x': r2_t[0],
                'int_y': Beta_t[0,1], 'slope_tx_y': Beta_t[1,1], 'slope_ty_y': Beta_t[2,1], 'slope_tz_y': Beta_t[3,1], 'R2_y': r2_t[1],
                'int_z': Beta_t[0,2], 'slope_tx_z': Beta_t[1,2], 'slope_ty_z': Beta_t[2,2], 'slope_tz_z': Beta_t[3,2], 'R2_z': r2_t[2],
            })
        pd.DataFrame(trans_rows).to_csv(
            os.path.join(args.out_dir, 'per_marker_local_translation_regression.csv'),
            index=False
        )
        print("Saved per-marker local translation regression to 'per_marker_local_translation_regression.csv'")
        # Visualization for translation: scatter of norm
        df_tr['norm_pred_t'] = np.linalg.norm(df_tr[['tx_pred','ty_pred','tz_pred']].to_numpy(), axis=1)
        df_tr['norm_gt_t']   = np.linalg.norm(df_tr[['tx_local','ty_local','tz_local']].to_numpy(), axis=1)
        err_nt = df_tr['norm_pred_t'] - df_tr['norm_gt_t']
        me_nt = err_nt.mean(); se_nt = err_nt.std()
        ss_res_nt = np.sum((df_tr['norm_gt_t'] - df_tr['norm_pred_t'])**2)
        ss_tot_nt = np.sum((df_tr['norm_gt_t'] - df_tr['norm_gt_t'].mean())**2)
        r2_nt = 1 - (ss_res_nt / ss_tot_nt) if ss_tot_nt > 0 else float('nan')
        plt.figure(figsize=(6,6))
        cmap = plt.get_cmap('tab10')
        for idx, mid in enumerate(sorted(df_tr['marker_id'].unique())):
            sel = df_tr['marker_id'] == mid
            plt.scatter(
                df_tr.loc[sel, 'norm_pred_t'],
                df_tr.loc[sel, 'norm_gt_t'],
                color=cmap(idx % 10), label=f'M{mid}', s=10, alpha=0.7
            )
        mnv_t = min(df_tr['norm_pred_t'].min(), df_tr['norm_gt_t'].min())
        mxv_t = max(df_tr['norm_pred_t'].max(), df_tr['norm_gt_t'].max())
        plt.plot([mnv_t, mxv_t], [mnv_t, mxv_t], 'k--', linewidth=1)
        plt.xlabel('Pred ∥t_local∥')
        plt.ylabel('GT   ∥t_local∥')
        plt.title(f"Local Translation Norm (R²={r2_nt:.3f}, μ_err={me_nt:.3f}, σ_err={se_nt:.3f})")
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
        plt.tight_layout()
        out_scatter_t = os.path.join(args.out_dir, 'local_translation_norm_scatter.png')
        plt.savefig(out_scatter_t)
        plt.close()
        print(f"Saved local translation norm scatter to '{out_scatter_t}'")

        # --- Angle Regression for Local ---
        df_gt_local_euler = df_gt[['frame','marker_id','rloc_x','rloc_y','rloc_z','input_angle']]
        df_gt_local_euler = df_gt_local_euler.drop_duplicates(['frame','marker_id']).reset_index(drop=True)
        def rvecs_to_euler_cols(rvecs):
            rolls, pitches, yaws = [], [], []
            for r in rvecs:
                R, _ = cv2.Rodrigues(r.astype(np.float32))
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
            return np.array(rolls), np.array(pitches), np.array(yaws)
        rvecs_local = df_gt_local_euler[['rloc_x','rloc_y','rloc_z']].to_numpy()
        ro_gt, pi_gt, ya_gt = rvecs_to_euler_cols(rvecs_local)
        df_gt_local_euler['roll_gt']  = ro_gt
        df_gt_local_euler['pitch_gt'] = pi_gt
        df_gt_local_euler['yaw_gt']   = ya_gt

        df_pred_local_euler = df_pred_local[['frame','marker_id','roll','pitch','yaw']].rename(
            columns={'roll':'roll_pred','pitch':'pitch_pred','yaw':'yaw_pred'}
        )
        df_merge_angle = pd.merge(df_gt_local_euler, df_pred_local_euler, on=['frame','marker_id'])

        rows_angle = []
        for mid in sorted(df_merge_angle['marker_id'].unique()):
            df_am = df_merge_angle[df_merge_angle['marker_id'] == mid].copy()
            X_a = df_am[['roll_pred','pitch_pred','yaw_pred']].to_numpy()
            y_a = df_am['input_angle'].to_numpy()
            model_a = LinearRegression().fit(X_a, y_a)
            coef_a = model_a.coef_
            intercept_a = model_a.intercept_
            r2_a = model_a.score(X_a, y_a)
            rows_angle.append({
                'marker_id':    mid,
                'coef_roll':    coef_a[0],
                'coef_pitch':   coef_a[1],
                'coef_yaw':     coef_a[2],
                'intercept':    intercept_a,
                'R2':           r2_a
            })
            df_am['pred_angle'] = model_a.predict(X_a)
            df_am['error'] = df_am['pred_angle'] - df_am['input_angle']
            err_mean_a = df_am['error'].mean()
            err_std_a  = df_am['error'].std()
            plt.figure(figsize=(8,4))
            plt.plot(df_am['frame'], df_am['input_angle'], 'r-', label='GT')
            plt.plot(df_am['frame'], df_am['pred_angle'],  'g-', label='Pred')
            plt.xlabel('Frame')
            plt.ylabel('Input Angle (deg')
            plt.title(f"M{mid}: Angle over Frames (R²={r2_a:.3f}, μ_err={err_mean_a:.3f}, σ_err={err_std_a:.3f})")
            plt.legend()
            plt.tight_layout()
            apath = os.path.join(args.out_dir, f"per_marker_{mid}_angle_timeseries.png")
            plt.savefig(apath)
            plt.close()
            print(f"Saved angle time-series for marker {mid} to '{apath}'")
        pd.DataFrame(rows_angle).to_csv(
            os.path.join(args.out_dir,'per_marker_local_angle_regression.csv'), index=False
        )
        print("Saved per-marker local angle regression to 'per_marker_local_angle_regression.csv'")
