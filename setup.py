# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import setuptools
import os
import re
import shutil
import warnings
from pathlib import Path

# Print an error message if PyTorch/torch_musa are not installed.
try:
    import torch
    import torch_musa
    from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension
except ImportError:
    # This happens if the user runs 'pip install' with default build isolation
    # OR if they simply don't have torch/torch_musa installed at all.
    print("\n\n" + "*" * 70)
    print("ERROR! Cannot compile nvdiffrast MUSA extension. Please ensure that:\n")
    print("1. You have PyTorch and torch_musa installed")
    print("2. You run 'pip install' with --no-build-isolation flag")
    print("*" * 70 + "\n\n")
    exit(1)

def _arch_as_int(arch):
    text = str(arch)
    if text.startswith("mp_"):
        text = text[3:]
    return int(text)


def _normalize_arch_list(arch_list):
    return sorted({str(_arch_as_int(arch.strip())) for arch in arch_list.split(";") if arch.strip()})


def _default_musa_arch_list():
    """Choose MUSA archs for extension builds when the user did not specify one."""
    try:
        device_count = torch.musa.device_count()
    except Exception:
        device_count = 0

    if device_count > 0:
        archs = set()
        supported = [_arch_as_int(arch) for arch in torch.musa.get_arch_list()]
        max_supported = max((arch // 10, arch % 10) for arch in supported) if supported else None
        for device_idx in range(device_count):
            capability = torch.musa.get_device_capability(device_idx)
            if max_supported is not None:
                capability = min(max_supported, capability)
            archs.add(f"{capability[0]}{capability[1]}")
        if archs:
            return ";".join(sorted(archs))

    # Building in a container without visible cards is common. Compile the
    # mainstream torch_musa archs instead of guessing one card generation.
    warnings.warn(
        "No MUSA device was visible while building nvdiffrast; defaulting "
        "TORCH_MUSA_ARCH_LIST to 21;22;31;32. Override TORCH_MUSA_ARCH_LIST "
        "if the target machine needs a narrower/different set.",
        RuntimeWarning,
    )
    return "21;22;31;32"


if "TORCH_MUSA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_MUSA_ARCH_LIST"] = _default_musa_arch_list()


def _embedded_musa_archs(path):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return []
    return sorted({match.decode("ascii") for match in re.findall(rb"musa-mtgpu-mt-musa--mp_([0-9]+)", data)})


def _remove_stale_arch_build_artifacts():
    expected = set(_normalize_arch_list(os.environ["TORCH_MUSA_ARCH_LIST"]))
    if not expected:
        return

    root = Path(__file__).resolve().parent
    stale_dirs = []
    stale_files = []

    for ninja_file in root.glob("build/temp.*/build.ninja"):
        try:
            text = ninja_file.read_text()
        except OSError:
            continue
        built = set(re.findall(r"--offload-arch=mp_([0-9]+)", text))
        if built and built != expected:
            stale_dirs.append(ninja_file.parent)

    for so_file in list(root.glob("_nvdiffrast_c*.so")) + list(root.glob("build/lib.*/_nvdiffrast_c*.so")):
        built = set(_embedded_musa_archs(so_file))
        if built and not expected.issubset(built):
            stale_files.append(so_file)

    if not stale_dirs and not stale_files:
        return

    warnings.warn(
        "Removing stale nvdiffrast MUSA build artifacts with offload archs that "
        f"do not match TORCH_MUSA_ARCH_LIST={';'.join(sorted(expected))}.",
        RuntimeWarning,
    )
    for path in stale_dirs:
        shutil.rmtree(path, ignore_errors=True)
    for path in stale_files:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class NvdiffrastBuildExtension(BuildExtension):
    def run(self):
        _remove_stale_arch_build_artifacts()
        super().run()


setuptools.setup(
    ext_modules=[
        MUSAExtension(
            "_nvdiffrast_c",
            sources=[
                "csrc/common/common.cpp",
                "csrc/common/cudaraster/impl/Buffer.cpp",
                "csrc/common/cudaraster/impl/CudaRaster.cpp",
                "csrc/common/cudaraster/impl/RasterImpl.cpp",
                "csrc/common/cudaraster/impl/RasterImpl_kernel.mu",
                "csrc/common/interpolate.mu",
                "csrc/common/rasterize.mu",
                "csrc/common/texture.cpp",
                "csrc/common/texture_kernel.mu",
                "csrc/torch/torch_bindings.cpp",
                "csrc/torch/torch_interpolate.cpp",
                "csrc/torch/torch_rasterize.cpp",
                "csrc/torch/torch_texture.cpp",
            ],
            extra_compile_args={
                "cxx": ["-DNVDR_TORCH"]
                + (["/wd4067", "/wd4624", "/wd4996"] if os.name == "nt" else []),
                "mcc": ["-DNVDR_TORCH"],
            },
        )
    ],
    cmdclass={"build_ext": NvdiffrastBuildExtension},
)
