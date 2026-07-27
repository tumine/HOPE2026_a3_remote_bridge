# Rockchip sysroot

Generate the untracked sysroot archive before cross-compiling:

```bash
bash scripts/export_rockchip_sysroot.sh
```

Expected output:

```text
thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz
```

The archive contains target ROS Jazzy headers/libraries and aarch64 system
development files. The `.sha256` file alone is not a sysroot.
