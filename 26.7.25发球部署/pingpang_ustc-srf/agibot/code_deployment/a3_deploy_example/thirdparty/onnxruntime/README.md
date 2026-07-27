# ONNX Runtime aarch64

This directory stores the prebuilt ONNX Runtime package used for RK3588
cross/arm64 builds of A3 deployment. x86 builds use the system
`find_package(onnxruntime)` result by default; the bundled archive is selected
only for `aarch64|arm64` when `USE_BUNDLED_ONNXRUNTIME_AARCH64=ON`.

Source:
`https://artifactory.infra.agibot.com/artifactory/third_party-local/onnxruntime/1.19.2/onnxruntime_gpu_1.19.2_aarch64.tar.gz`

Expected SHA256:
`c3f7dd6dabd37551ad663640ba51c7aa2a651f43f5241c1499de0735aebeb026`

Archive root after extraction:
`onnxruntime_gpu_1.19.2/`
