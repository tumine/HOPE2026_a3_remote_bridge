# A3 joint order

One canonical 31-DOF joint order is shared by training, the exported ONNX
policy, and the deploy runner. The source of truth is
[`hope_training/config/joint_order_agibot_a3.yaml`](../../hope_training/config/joint_order_agibot_a3.yaml);
the deploy runner carries the same list in
[`a3_deploy_onnx_ref_pingpong/joint_order.py`](../../a3_deploy/a3_deploy_example/reference/a3_deploy_onnx_ref_pingpong/joint_order.py).
The authoritative contract (with the full indexed table) is
[POLICY_INTERFACE.md — Joint order](../POLICY_INTERFACE.md#joint-order).

The observation `joint_pos` / `joint_vel` / `last_action` slices, the ONNX
`raw_action[31]` output, and the joint-position targets written to the robot all
use this exact index order:

```text
 0  waist_yaw_joint              16  right_wrist_roll_joint
 1  waist_roll_joint             17  right_wrist_pitch_joint
 2  waist_pitch_joint            18  right_wrist_yaw_joint
 3  head_yaw_joint    (passive)  19  left_hip_pitch_joint
 4  head_pitch_joint  (passive)  20  left_hip_roll_joint
 5  left_shoulder_pitch_joint    21  left_hip_yaw_joint
 6  left_shoulder_roll_joint     22  left_knee_joint
 7  left_shoulder_yaw_joint      23  left_ankle_pitch_joint
 8  left_elbow_joint             24  left_ankle_roll_joint
 9  left_wrist_roll_joint        25  right_hip_pitch_joint
10  left_wrist_pitch_joint       26  right_hip_roll_joint
11  left_wrist_yaw_joint         27  right_hip_yaw_joint
12  right_shoulder_pitch_joint   28  right_knee_joint
13  right_shoulder_roll_joint    29  right_ankle_pitch_joint
14  right_shoulder_yaw_joint     30  right_ankle_roll_joint
15  right_elbow_joint
```

- `head_yaw_joint` / `head_pitch_joint` (indices **3–4**) are passive at deploy:
  held at their default angle, zeroed in the applied action, but still occupying
  their action columns so every vector is length 31.
- The racket is mounted on the right wrist.

## Enforcement

Isaac/PhysX may enumerate an articulation in kinematic-tree order rather than
the semantic order above. That internal order is not part of the public policy
contract. Training resolves every joint-bearing policy boundary by exact name:

- `joint_pos` and `joint_vel` are gathered into canonical order;
- action columns are canonical and are scattered to PhysX joint ids;
- reference-motion joint streams and reset states remain canonical;
- randomized default positions are gathered back into canonical action columns.

The name mapping and public order are checked at three points:

| Stage | Check |
|-------|-------|
| Train (`scripts/train.py`) | Requires exact canonical joint-name coverage and verifies that action columns resolve in canonical order; logs the canonical-to-PhysX index mapping. |
| Export (`scripts/export_onnx.py`) | Applies the same mapping checks, then embeds the canonical order in the ONNX metadata and `policy_manifest.json`. |
| Deploy (reference runner) | Rejects any `hope_pingpong.onnx` whose embedded `joint_order` metadata differs from the runner's canonical list. |

An asset may use a different internal enumeration as long as it contains the
same 31 unique joint names. Do not change the canonical YAML to match a private
simulator order. Verify the canonical list against the real robot before
deploying, e.g. `ros2 topic echo /joint_states --once`.
