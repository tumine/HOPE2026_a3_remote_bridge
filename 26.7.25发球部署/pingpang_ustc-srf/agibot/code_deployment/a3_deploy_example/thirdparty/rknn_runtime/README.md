# RKNN Runtime Bundle

This directory vendors the Rockchip RKNN runtime files needed to build and run
the A3 deploy RKNN backend without relying on target-board development headers.

Version pinned here: `2.3.2`, matching the A3 RK3588 boards currently reporting:

```text
librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
RKNPU driver: v0.9.8
```

Source:

```text
https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2/rknpu2/runtime/Linux/librknn_api
```

Bundled files:

```text
2.3.2/include/rknn_api.h
2.3.2/include/rknn_matmul_api.h
2.3.2/include/rknn_custom_op.h
2.3.2/lib/aarch64/librknnrt.so
```

Known SHA256:

```text
c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a  rknn_api.h
aaadd9a7118de30a06b222996b6731db77095d00f5931a7a98c83a67f14a4d42  rknn_matmul_api.h
af5983da0ca244ca31dc3162aa683322b0285531196c7a770f29cd2e3b8ccaa6  rknn_custom_op.h
d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8  librknnrt.so
```
