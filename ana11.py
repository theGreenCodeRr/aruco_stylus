import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === File paths (adjust as needed) ===
GT_CSV     = '6DoF_annotated_poses.csv'
LOCAL_CSV  = 'cam0_aruco_local.csv'
GLOBAL_CSV = 'cam0_aruco_global.csv'
OUT_CSV    = 'cam0_best_local.csv'

# === 1. Load and reverse-map ground truth ===
df_gt = pd.read_csv(GT_CSV)
max_frame = df_gt['frame'].max()
df_gt['frame'] = max_frame - df_gt['frame']  # reverse frame-to-angle mapping
df_gt = df_gt[['frame','input_angle','marker_id']]

# === 2. Load local & global estimates ===
df_local  = pd.read_csv(LOCAL_CSV)
df_global = pd.read_csv(GLOBAL_CSV)

# === 3. Merge GT angles (and marker_id for global) ===
df_local  = df_local .merge(df_gt[['frame','input_angle']], on='frame', how='left')
df_global = df_global.merge(df_gt[['frame','input_angle','marker_id']], on='frame', how='left')

# === 4. Compute absolute yaw error ===
df_local ['err'] = np.abs(df_local ['yaw'] - df_local ['input_angle'])
df_global['err'] = np.abs(df_global['yaw'] - df_global['input_angle'])

# === 5. Best local marker per degree ===
best_local = (
    df_local
    .loc[df_local.groupby('input_angle')['err'].idxmin()]
    .sort_values('input_angle')
)
# Export best-local for comparison
best_local.to_csv(OUT_CSV, index=False)

# === 6. Top-2 global markers per degree ===
dfg = df_global.sort_values(['input_angle','err'])
dfg['rank'] = dfg.groupby('input_angle')['err'].rank(method='first')
best2_global = dfg[dfg['rank'] <= 2]

# === 7. Plotting ===
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 7.1 Local best IDs
axes[0].plot(best_local['input_angle'], best_local['marker_id'],
             'o-', color='orange', label='Best Local')
axes[0].set_ylabel('Marker ID')
axes[0].set_title('Local: Best Marker per Degree (Min Rotation Error)')
axes[0].grid(True)
axes[0].set_xlim(0, 120)
axes[0].set_ylim(best_local['marker_id'].min()-1,
                 best_local['marker_id'].max()+1)

# 7.2 Global top-2 IDs
colors = {1: 'blue', 2: 'green'}
for r in [1, 2]:
    sub = best2_global[best2_global['rank'] == r]
    axes[1].plot(sub['input_angle'], sub['marker_id'],
                 's-', color=colors[r], label=f'Global Rank {int(r)}')
axes[1].set_xlabel('Ground Truth Angle (°)')
axes[1].set_ylabel('Marker ID')
axes[1].set_title('Global: Top-2 Markers per Degree')
axes[1].legend()
axes[1].grid(True)
axes[1].set_xlim(0, 120)
axes[1].set_ylim(best2_global['marker_id'].min()-1,
                 best2_global['marker_id'].max()+1)

plt.tight_layout()
plt.show()
