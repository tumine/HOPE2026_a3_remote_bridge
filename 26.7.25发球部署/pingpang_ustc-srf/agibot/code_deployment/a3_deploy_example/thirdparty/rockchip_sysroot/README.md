# Rockchip AArch64 Sysroot Bundle

Rockchip deployment builds use an x86_64 builder image plus a local copy of the
Rockchip arm64 target sysroot. The build no longer needs to run
`registry.agibot.com/agibot-tech/rockchip:1.0` through qemu during normal
package builds.

Expected bundle:

```text
a3_deploy_example/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz
```

Generate or refresh it once with:

```bash
scripts/export_rockchip_sysroot.sh
```

The tarball contains the directories that `Dockerfile.a3-rockchip-builder`
restores into the builder image:

```text
opt/ros/jazzy
usr/include
usr/share/eigen3
usr/lib/aarch64-linux-gnu
```

If the Rockchip target image changes, regenerate this tarball from the matching
image and keep the `.sha256` file beside it.
