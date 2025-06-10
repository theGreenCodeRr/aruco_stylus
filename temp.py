import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === File paths (adjust as needed) ===
GT_CSV    = '6DoF_annotated_poses.csv'   # Ground-truth annotations
GLOBAL_CSV = 'cam0_aruco_global.csv'     # ArUco global pose estimates
OUT_CSV    = 'cam0_aruco_global_regression.csv'  # Regression results

# === 1. Load ground truth angles ===
df_gt = pd.read_csv(GT_CSV)[['frame', 'input_angle']]

# === 2. Load global pose estimates and merge GT angles ===
df_global = pd.read_csv(GLOBAL_CSV)
df = df_global.merge(df_gt, on='frame', how='left')

# === 3. Drop any rows without GT ===
df = df.dropna(subset=['input_angle'])

# === 4. Prepare regression data ===
# X = measured ArUco yaw, y = true input angle
y = df['input_angle'].values.reshape(-1,1)
X = df['yaw'].values.reshape(-1,1)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)
r2 = model.score(X, y)

# Add predictions to DataFrame
df['pred_angle'] = model.predict(X).flatten()

# Export regression results
df[['frame','yaw','input_angle','pred_angle']].to_csv(OUT_CSV, index=False)

# === 5. Plot scatter + regression line with inverted GT axis ===
plt.figure(figsize=(8,6))
# scatter of ArUco yaw vs GT angle
plt.scatter(df['yaw'], df['input_angle'], alpha=0.6, label='Data')

# regression fit line
x_line = np.linspace(df['yaw'].min(), df['yaw'].max(), 200).reshape(-1,1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, 'r-', label=f'Regression (R²={r2:.2f})')

# ideal diagonal
diag = np.linspace(0,120,100)
plt.plot(diag, diag, 'k--', label='Ideal')

plt.xlabel('ArUco Global Yaw (°)')
plt.ylabel('Ground Truth Angle (°)')
plt.title('Plain ARUCO Angle Regression')
plt.xlim(0,120)
plt.ylim(120,0)  # invert y-axis to go from 120 down to 0
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
