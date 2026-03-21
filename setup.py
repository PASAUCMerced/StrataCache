# SPDX-License-Identifier: Apache-2.0
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# 路径配置
ROOT_DIR = Path(__file__).parent
PCM_DIR = os.path.abspath("/workspace/projects/pcm")
PCM_LIB_PATH = os.path.join(PCM_DIR, "build", "src", "libPCM_STATIC_SILENT.a")

class PCMBuildExt(build_ext):
    """自定义构建类：先用 CMake 构建 PCM 静态库，再编译 Python 扩展"""
    def run(self):
        # 1. 自动构建 PCM 静态库 (基于 CMake)
        if not os.path.exists(PCM_LIB_PATH):
            print(f"--- PCM static library not found. Building at {PCM_DIR} ---")
            build_dir = os.path.join(PCM_DIR, "build")
            os.makedirs(build_dir, exist_ok=True)

            subprocess.check_call(["cmake", ".."], cwd=build_dir)
            subprocess.check_call(
                [
                    "cmake",
                    "--build", ".",
                    "--config", "Release",
                    "--parallel",
                    "--target", "PCM_STATIC",
                ],
                cwd=build_dir,
            )

        # 2. 调用原生 build_ext 逻辑
        super().run()

def get_pybind_include():
    """获取 pybind11 头文件路径"""
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        print("Error: pybind11 is required to build this extension.")
        sys.exit(1)

def get_extensions():
    # 移除 torch 相关的 ABI 标志，使用标准的编译器参数
    # 定义宏（保留你需要的宏）
    define_macros = [("__HIP_PLATFORM_HCC__", "1"), ("USE_ROCM", "1")]

    extensions = [
        Extension(
            name="stratacache.pcm",
            # 包含你所有的源文件
            sources=["src/stratacache/csrc/pcm.cpp", "src/stratacache/csrc/pybind.cpp"],
            include_dirs=[
                os.path.join(PCM_DIR, "src"),
                "src/stratacache/csrc",
                get_pybind_include(),  # 关键：手动包含 pybind11
            ],
            extra_objects=[PCM_LIB_PATH],  # 静态链接 libpcm.a
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
            libraries=["pthread", "rt"],
            define_macros=define_macros,
            language="c++",
        ),
    ]
    return extensions

if __name__ == "__main__":
    setup(
        name="stratacache",
        packages=find_packages(exclude=("csrc",)),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": PCMBuildExt},
        include_package_data=True,
    )