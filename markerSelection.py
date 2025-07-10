import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load GT (6DoF) for ground-truth angles
df_gt = pd.read_csv('CSVs/6DoF_annotated_poses.csv')
max_frame = df_gt['frame'].max()
df_gt['frame'] = max_frame - df_gt['frame']
df_gt = df_gt[['frame', 'input_angle']]

# Load local detections
df_local = pd.read_csv('CSVs/aruco_local.csv')

# Merge ground truth angles into local detections
df_local = df_local.merge(df_gt, on='frame', how='left')

# Compute absolute yaw error and select best marker per angle
df_local['err'] = np.abs(df_local['yaw'] - df_local['input_angle'])
best_local = df_local.loc[df_local.groupby('input_angle')['err'].idxmin()].sort_values('input_angle')

# ——— NEW: fixed marker ID list for consistent colors across runs ———
fixed_marker_ids = sorted(df_local['marker_id'].unique())  # replace with full list if known
cmap = plt.get_cmap('tab10')
color_map = {mid: cmap(mid % cmap.N) for mid in fixed_marker_ids}
# ————————————————————————————————————————————————————————————

# Plot setup
plt.rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top subplot: selected (best) marker per angle
axes[0].text(0.98, 0.02, 'selected_marker', transform=axes[0].transAxes,
             fontsize=18, fontweight='bold', ha='right', va='bottom')
axes[0].plot(best_local['input_angle'], best_local['marker_id'], '-', linewidth=2, alpha=0.6)
for mid, grp in best_local.groupby('marker_id'):
    cnt = len(grp)
    axes[0].plot(grp['input_angle'], grp['marker_id'], 'o',
                 color=color_map[mid],  # consistent color
                 label=f'ID {mid} ({cnt})')
axes[0].set_ylabel('Marker ID')
axes[0].grid(True)
axes[0].set_xlim(0, 120)
axes[0].set_ylim(best_local['marker_id'].min() - 1,
                 best_local['marker_id'].max() + 1)

# Bottom subplot: all marker detections
axes[1].text(0.98, 0.02, 'all_markers', transform=axes[1].transAxes,
             fontsize=18, fontweight='bold', ha='right', va='bottom')
for mid, grp in df_local.groupby('marker_id'):
    cnt = len(grp)
    axes[1].plot(grp['input_angle'], grp['marker_id'], 's',
                 color=color_map[mid],  # consistent color
                 label=f'ID {mid} ({cnt})')
axes[1].set_xlabel('Ground Truth Angle (°)')
axes[1].set_ylabel('Marker ID')
axes[1].grid(True)
axes[1].set_xlim(0, 120)
axes[1].set_ylim(df_local['marker_id'].min() - 1,
                 df_local['marker_id'].max() + 1)

plt.tight_layout()
plt.show()
