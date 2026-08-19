# Waist log tracking-error summary

- Parsed waist_diag samples: 217
- Active motion samples used: 123
- Active data span: 122.000 s
- Median log interval: 1.000 s
- Position error definition: q_exec - q_feedback
- Torque discrepancy definition: tau_theoretical - tau_feedback

| Joint | Pos MAE (deg) | Pos RMSE (deg) | Pos P95 (deg) | Pos max (deg @ s) | Torque MAE (N m) | Torque RMSE (N m) | Torque P95 (N m) | Torque max (N m @ s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| waist_yaw | 1.259 | 3.185 | 8.136 | 21.371 @ 120.0 | 1.311 | 3.542 | 5.952 | 22.617 @ 111.0 |
| waist_roll | 4.947 | 6.715 | 14.209 | 26.814 @ 119.0 | 0.533 | 1.257 | 2.992 | 7.529 @ 110.0 |
| waist_pitch | 2.205 | 4.345 | 10.371 | 17.303 @ 96.0 | 0.883 | 2.217 | 4.938 | 15.176 @ 111.0 |

The 1 Hz status log is a downsampled snapshot of the 50 Hz control loop; transient peaks may be missed.
