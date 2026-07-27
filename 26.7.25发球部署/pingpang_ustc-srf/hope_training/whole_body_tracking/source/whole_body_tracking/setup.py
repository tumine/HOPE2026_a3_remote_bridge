"""Installation script for the ``whole_body_tracking`` Isaac Lab extension."""

import os

import toml
from setuptools import setup

# Read the extension metadata (single source of truth for version / author / description).
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum runtime dependencies (Isaac Lab itself is provided by the base install).
INSTALL_REQUIRES = [
    "psutil",
    # Keep export dependencies compatible with Isaac Lab 2.1.x.
    "onnx==1.16.1",
    "onnxscript==0.1.0",
    "onnxruntime>=1.18,<2",
    "mujoco>=3.1,<4",
    "pyyaml",
    "tensorboard>=2.10",
    # Isaac Lab 2.1.x integrates against this runner version.
    "rsl-rl-lib==2.3.3",
]

setup(
    name="whole_body_tracking",
    packages=["whole_body_tracking"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.0.0",
    ],
    zip_safe=False,
)
