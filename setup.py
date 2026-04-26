# SPDX-License-Identifier: Apache-2.0
"""
StrataCache build script.

By default this builds a pure-Python install: no C/C++ extensions, no
PCM dependency, no CMake. The optional `stratacache.pcm` Intel-PCM
binding (used by the CPU-side memory/PCIe bandwidth telemetry) is
opt-in:

    STRATACACHE_BUILD_PCM=1 PCM_DIR=/path/to/pcm pip install -e .

When opted in, PCM_DIR must point at a checkout of Intel PCM and the
static library `build/src/libPCM_STATIC_SILENT.a` must already be built
(or be buildable via `cmake .. && cmake --build . --target PCM_STATIC`
in `${PCM_DIR}/build`). When opted out (the default), the cpu-telemetry
module degrades gracefully (telemetry init logs a warning and skips
PCM-derived metrics).

The CXL backend's libcxl_shm.so is built independently via
`stratacache/csrc/cxl/Makefile` and is not handled here.
"""
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent

# ---- Optional PCM extension --------------------------------------------------
BUILD_PCM = os.environ.get("STRATACACHE_BUILD_PCM", "").strip() in ("1", "true", "yes")
PCM_DIR = os.path.abspath(os.environ.get("PCM_DIR", ""))
PCM_LIB_PATH = (
    os.path.join(PCM_DIR, "build", "src", "libPCM_STATIC_SILENT.a")
    if PCM_DIR
    else ""
)


class PCMBuildExt(build_ext):
    """Build the PCM static library on demand, then the Python extension."""

    def run(self):
        if BUILD_PCM and PCM_LIB_PATH and not os.path.exists(PCM_LIB_PATH):
            if not PCM_DIR or not os.path.isdir(PCM_DIR):
                raise RuntimeError(
                    "STRATACACHE_BUILD_PCM=1 but PCM_DIR is unset or invalid; "
                    "set PCM_DIR to a checkout of Intel PCM."
                )
            print(f"--- PCM static library not found. Building at {PCM_DIR} ---")
            build_dir = os.path.join(PCM_DIR, "build")
            os.makedirs(build_dir, exist_ok=True)
            subprocess.check_call(["cmake", ".."], cwd=build_dir)
            subprocess.check_call(
                [
                    "cmake",
                    "--build",
                    ".",
                    "--config",
                    "Release",
                    "--parallel",
                    "--target",
                    "PCM_STATIC",
                ],
                cwd=build_dir,
            )
        super().run()


def _get_pybind_include() -> str:
    try:
        import pybind11

        return pybind11.get_include()
    except ImportError:
        print("pybind11 is required to build the optional PCM extension.")
        sys.exit(1)


def get_extensions():
    if not BUILD_PCM:
        return []
    if not PCM_DIR:
        raise RuntimeError(
            "STRATACACHE_BUILD_PCM=1 requires PCM_DIR to be set."
        )
    return [
        Extension(
            name="stratacache.pcm",
            sources=[
                "src/stratacache/csrc/pcm.cpp",
                "src/stratacache/csrc/pybind.cpp",
            ],
            include_dirs=[
                os.path.join(PCM_DIR, "src"),
                "src/stratacache/csrc",
                _get_pybind_include(),
            ],
            extra_objects=[PCM_LIB_PATH],
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
            libraries=["pthread", "rt"],
            language="c++",
        )
    ]


if __name__ == "__main__":
    setup(
        name="stratacache",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=get_extensions(),
        cmdclass={"build_ext": PCMBuildExt} if BUILD_PCM else {},
        include_package_data=True,
    )
