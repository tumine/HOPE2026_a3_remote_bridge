# Policy models

The current deployment actor is included here:

```
models/hope_pingpong.onnx      # obs[1,111] -> raw_action[1,31], single output
models/policy_manifest.json    # contract, layout, joint order, and source checkpoint
```

The reference runtime config (`../config/hope_pingpong_runtime.yaml`) points
`policy.onnx_path` at `../models/hope_pingpong.onnx`. Override on the command line
with `--onnx /path/to/hope_pingpong.onnx`.

Bundled source:

- run: `2026-07-26_01-47-04`
- checkpoint: `model_21500.pt`
- checkpoint SHA256:
  `7939b3a2e4716e764271126668602b0826f6f4fc9615a74ccdb7a0761ee19987`
- ONNX SHA256:
  `6def579da350d75d53f733ad6ba742336738b4d18267f111a2d33d2d8dcadf09`

Export a replacement with the training package's `export_onnx.py`, then replace
both files together. A replacement policy must retain the same 111-D/31-D
contract or bring its own matching runtime, ActionAdapter, and documentation.
