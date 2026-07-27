#!/usr/bin/env python3
"""Prepare (or verify) the Agibot A3 ping-pong URDF as an Isaac Lab asset.

The starter ships the Agibot-provided A3 ping-pong URDF package under
``agibot/URDF/A3T2.5-URDF-std-pingpang/`` (vendor material, no OSS license — see ``A3_ASSETS.md``),
and this script uses it as the default source. To use your OWN vendor-supplied copy instead, place
it under ``a3_deploy/URDF/`` (see ``a3_deploy/URDF/README.md``) in this layout and point
``--source-root`` at it::

    a3_deploy/URDF/<your_a3_package>/
        urdf/<robot>.urdf
        meshes/*.STL

This script copies the meshes and rewrites the URDF's ``package://.../meshes/*.STL`` references to
repo-relative ``../meshes/*.STL`` so Isaac Lab can load the model without ROS package resolution.
The generated asset is written under the training package's ``assets/agibot_a3`` directory
(git-ignored; regenerate it locally).

Usage (from the repo root):
    python3 hope_training/whole_body_tracking/scripts/prepare_a3_isaac_asset.py --force
    python3 hope_training/whole_body_tracking/scripts/prepare_a3_isaac_asset.py \\
        --source-root a3_deploy/URDF/<your_a3_package>   # use your own copy
    python3 hope_training/whole_body_tracking/scripts/prepare_a3_isaac_asset.py --check
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import xml.etree.ElementTree as ET


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir() and (parent / "a3_deploy").is_dir():
            return parent
    return here.parents[3]


REPO_ROOT = _repo_root()
# Source URDF package: the Agibot-provided A3 ping-pong package shipped with the starter under
# agibot/URDF/ when present, else a user-supplied copy under a3_deploy/URDF/ (see its README).
_SOURCE_CANDIDATES = (
    REPO_ROOT / "agibot" / "URDF" / "A3T2.5-URDF-std-pingpang",
    REPO_ROOT / "a3_deploy" / "URDF" / "agibot_a3_pingpong",
)
DEFAULT_SOURCE_ROOT = next((p for p in _SOURCE_CANDIDATES if p.is_dir()), _SOURCE_CANDIDATES[0])
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
    / "agibot_a3"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_SOURCE_ROOT,
                        help="Your A3 URDF package root (contains urdf/ and meshes/).")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Where to write the prepared Isaac asset.")
    parser.add_argument("--force", action="store_true", help="Recreate the output root before copying.")
    parser.add_argument("--check", action="store_true", help="Only verify the already-prepared output asset.")
    return parser.parse_args()


def _rewrite_mesh_ref(filename: str) -> str:
    """``package://<pkg>/.../meshes/foo.STL`` -> ``../meshes/foo.STL`` (leave other refs untouched)."""
    if filename.startswith("package://") and "/meshes/" in filename:
        return "../meshes/" + filename.rsplit("/meshes/", 1)[1]
    return filename


def _remove_fixed_joint_axes(root: ET.Element) -> int:
    """Remove meaningless fixed-joint axes that Isaac's URDF importer rejects.

    The vendor URDF contains fixed joints with empty, scalar, or zero-vector
    ``axis`` attributes. URDF does not require an axis for a fixed joint, so
    dropping those elements preserves the kinematics and avoids importer errors.
    """
    removed = 0
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        axis = joint.find("axis")
        if axis is not None:
            joint.remove(axis)
            removed += 1
    return removed


def _find_source_urdf(source_root: pathlib.Path) -> pathlib.Path:
    urdf_dir = source_root / "urdf"
    candidates = sorted(urdf_dir.glob("*.urdf")) if urdf_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"no .urdf found under {urdf_dir}. Provide your own A3 URDF package "
            "(see a3_deploy/URDF/README.md)."
        )
    return candidates[0]


def prepare(source_root: pathlib.Path, output_root: pathlib.Path, force: bool) -> None:
    source_urdf = _find_source_urdf(source_root)
    source_meshes = source_root / "meshes"
    if not source_meshes.is_dir():
        raise FileNotFoundError(f"missing meshes directory: {source_meshes}")

    if output_root.exists() and force:
        shutil.rmtree(output_root)
    urdf_dir = output_root / "urdf"
    mesh_dir = output_root / "meshes"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_meshes, mesh_dir, dirs_exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", _rewrite_mesh_ref(filename))
    removed_axes = _remove_fixed_joint_axes(root)
    tree.write(urdf_dir / "model.urdf", encoding="utf-8", xml_declaration=True)
    print(
        f"[prepare_a3_isaac_asset] prepared: {output_root} "
        f"(removed {removed_axes} fixed-joint axis elements)"
    )


def check(output_root: pathlib.Path) -> bool:
    model_urdf = output_root / "urdf" / "model.urdf"
    if not model_urdf.exists():
        print(f"[FAIL] missing prepared URDF: {model_urdf}")
        return False

    refs = [m.get("filename") for m in ET.parse(model_urdf).getroot().findall(".//mesh") if m.get("filename")]
    failures: list[str] = []

    stale = [r for r in refs if r.startswith("package://") and "/meshes/" in r]
    if stale:
        failures.append("stale package:// mesh refs remain: " + ", ".join(sorted(set(stale))[:8]))

    rel_refs = [r for r in refs if r.startswith("../meshes/")]
    for ref in rel_refs:
        if not (model_urdf.parent / ref).resolve().exists():
            failures.append(f"missing mesh referenced by URDF: {ref}")

    root = ET.parse(model_urdf).getroot()
    fixed_axes = [
        joint.get("name", "<unnamed>")
        for joint in root.findall("joint")
        if joint.get("type") == "fixed" and joint.find("axis") is not None
    ]
    if fixed_axes:
        failures.append(
            "fixed joints still contain axis elements: " + ", ".join(fixed_axes[:8])
        )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return False
    print(f"[prepare_a3_isaac_asset] OK: {len(refs)} mesh refs ({len(rel_refs)} repo-relative), all resolve.")
    return True


def main() -> int:
    args = _parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not args.check:
        prepare(source_root, output_root, args.force)
    return 0 if check(output_root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
