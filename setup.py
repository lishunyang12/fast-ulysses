import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()


class CMakeExtension(Extension):
    def __init__(self):
        super().__init__("fast_ulysses._C", sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        output = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build = ROOT / "build"
        arch = os.environ.get("FAST_ULYSSES_CUDA_ARCH")
        if not arch:
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
        env = os.environ.copy()
        env["TORCH_CUDA_ARCH_LIST"] = f"{arch[:-1]}.{arch[-1]}"
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(build),
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DEXT_SUFFIX={sysconfig.get_config_var('EXT_SUFFIX') or '.so'}",
                f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            env=env,
        )
        subprocess.check_call(["cmake", "--build", str(build), "-j4"], env=env)


setup(
    name="fast-ulysses",
    version="0.3.0.dev0",
    packages=["fast_ulysses"],
    install_requires=["torch>=2.10"],
    ext_modules=[CMakeExtension()],
    cmdclass={"build_ext": CMakeBuild},
)
