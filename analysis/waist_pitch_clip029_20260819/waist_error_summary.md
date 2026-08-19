# Waist log tracking-error summary

- Parsed waist_diag samples: 175
- Active motion samples used: 108
- Active data span: 121.000 s
- Median log interval: 1.000 s
- Position error definition: q_exec - q_feedback
- Torque discrepancy definition: tau_theoretical - tau_feedback

| Joint | Pos MAE (deg) | Pos RMSE (deg) | Pos P95 (deg) | Pos max (deg @ s) | Torque MAE (N m) | Torque RMSE (N m) | Torque P95 (N m) | Torque max (N m @ s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| waist_yaw | 1.430 | 2.594 | 6.875 | 13.350 @ 121.0 | 1.010 | 2.547 | 3.992 | 18.245 @ 121.0 |
| waist_roll | 5.516 | 7.629 | 18.850 | 25.038 @ 52.0 | 0.864 | 1.979 | 4.230 | 11.151 @ 50.0 |
| waist_pitch | 7.838 | 7.989 | 10.943 | 16.960 @ 98.0 | 1.065 | 4.021 | 1.974 | 40.102 @ 121.0 |

The 1 Hz status log is a downsampled snapshot of the 50 Hz control loop; transient peaks may be missed.
