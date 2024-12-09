import numpy as np
import matplotlib.pyplot as plt

# Data
distances = [600, 500, 400, 300]  # mm

# Case 1: Stationary Camera and Chessboard
case1_focal_x = [2321.79, 876.72, 840.60, 823.95]
case1_focal_y = [2279.79, 878.58, 841.69, 806.99]
case1_principal_x = [320.44, 329.82, 339.84, 435.88]
case1_principal_y = [239.65, 264.77, 262.17, 255.48]
case1_distortion_k1 = [5.075, -0.965, -0.0962, 0.407]
case1_rms = [0.137, 0.126, 0.127, 0.154]

# Case 2: Moving Camera, Stationary Chessboard
case2_focal_x = [486.18, 658.33, 435.98, 498.22]
case2_focal_y = [489.21, 659.84, 438.16, 499.47]
case2_principal_x = [278.54, 318.80, 313.79, 331.00]
case2_principal_y = [276.81, 251.48, 278.78, 263.49]
case2_distortion_k1 = [0.206, 0.211, 0.117, 0.124]
case2_rms = [0.140, 0.158, 0.160, 0.225]

# Create plots
plt.figure(figsize=(15, 12))

# Focal length (f_x)
plt.subplot(3, 2, 1)
plt.plot(distances, case1_focal_x, marker='o', label="Case 1")
plt.plot(distances, case2_focal_x, marker='x', label="Case 2")
plt.title("Focal Length (f_x) vs Distance")
plt.xlabel("Distance (mm)")
plt.ylabel("Focal Length (pixels)")
plt.legend()

# Focal length (f_y)
plt.subplot(3, 2, 2)
plt.plot(distances, case1_focal_y, marker='o', label="Case 1")
plt.plot(distances, case2_focal_y, marker='x', label="Case 2")
plt.title("Focal Length (f_y) vs Distance")
plt.xlabel("Distance (mm)")
plt.ylabel("Focal Length (pixels)")
plt.legend()

# Principal point (c_x)
plt.subplot(3, 2, 3)
plt.plot(distances, case1_principal_x, marker='o', label="Case 1")
plt.plot(distances, case2_principal_x, marker='x', label="Case 2")
plt.title("Principal Point (c_x) vs Distance")
plt.xlabel("Distance (mm)")
plt.ylabel("Principal Point X (pixels)")
plt.legend()

# Principal point (c_y)
plt.subplot(3, 2, 4)
plt.plot(distances, case1_principal_y, marker='o', label="Case 1")
plt.plot(distances, case2_principal_y, marker='x', label="Case 2")
plt.title("Principal Point (c_y) vs Distance")
plt.xlabel("Distance (mm)")
plt.ylabel("Principal Point Y (pixels)")
plt.legend()

# Distortion coefficient (k1)
plt.subplot(3, 2, 5)
plt.plot(distances, case1_distortion_k1, marker='o', label="Case 1")
plt.plot(distances, case2_distortion_k1, marker='x', label="Case 2")
plt.title("Distortion Coefficient (k1) vs Distance")
plt.xlabel("Distance (mm)")
plt.ylabel("k1")
plt.legend()

# RMS error
plt.subplot(3, 2, 6)
plt.plot(distances, case1_rms, marker='o', label="Case 1")
plt.plot(distances, case2_rms, marker='x', label="Case 2")
plt.title("RMS Error vs Distance")
plt.xlabel("Distance (mm)")
plt.ylabel("RMS Error")
plt.legend()

plt.tight_layout()
plt.show()
