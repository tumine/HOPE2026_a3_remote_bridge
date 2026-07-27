# Thor AArch64 Sysroot Bundle

Thor deployment builds use the x86 DRIVE OS builder image plus a local copy of
the Thor arm64 target sysroot. The build no longer needs to pull
`registry.agibot.com/agibot-tech/devops/thor:1.0` during normal package builds.

Expected bundle:

```text
a3_deploy_example/thirdparty/thor_sysroot/thor-1.0-aarch64-sysroot.tar.gz
```

Generate or refresh it once with:

```bash
scripts/export_thor_sysroot.sh
```

The tarball contains the directories that `Dockerfile.a3-thor-builder` restores
into the builder image:

```text
opt/ros/jazzy
usr/include
usr/share/eigen3
usr/lib/aarch64-linux-gnu
```

If the Thor target image changes, regenerate this tarball from the matching
image and keep the `.sha256` file beside it.
