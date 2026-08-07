from __future__ import annotations

import os
import re
import shlex
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)


# Minimum NVSHMEM this extension is built against. 3.4.5 is what the nvidia-nvshmem-cu13
# wheel ships and what every API here was checked against; nothing below it has been tried.
_NVSHMEM_MIN = (3, 4, 5)


def _nvshmem_version(root: Path) -> tuple[int, int, int] | None:
    """(major, minor, patch) from the install's version header, or None if unreadable."""
    header = root / "include" / "non_abi" / "nvshmem_version.h"
    try:
        text = header.read_text()
    except OSError:
        return None
    parts = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"NVSHMEM_VENDOR_{field}_VERSION\s+(\d+)", text)
        if m is None:
            return None
        parts.append(int(m.group(1)))
    return tuple(parts)  # type: ignore[return-value]


def _nvshmem_candidates() -> list[tuple[str, Path]]:
    """(why, root) in preference order: explicit override, then torch's own wheel, then /usr.

    torch depends on nvidia-nvshmem-cu13, so on a machine that can run this extension the
    library is already installed and version-matched to torch. Using it means one fewer
    thing to install and no chance of loading a second NVSHMEM alongside torch's.
    """
    out: list[tuple[str, Path]] = []
    env = os.environ.get("NVSHMEM_HOME", "")
    if env:
        out.append(("NVSHMEM_HOME", Path(env)))
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        out.append(("torch's nvidia-nvshmem wheel", Path(site_dir) / "nvidia" / "nvshmem"))
    out.append(("system", Path("/usr")))
    return out


def _resolve_nvshmem() -> tuple[Path, Path]:
    """(root, host library). Reports every candidate and why it was rejected."""
    rejected = []
    for why, root in _nvshmem_candidates():
        if not (root / "include" / "nvshmem.h").is_file():
            rejected.append(f"  {why}: {root} -- no include/nvshmem.h")
            continue
        libs = sorted((root / "lib").glob("libnvshmem_host.so*")) or sorted(
            (root / "lib64").glob("libnvshmem_host.so*")
        )
        if not libs:
            rejected.append(f"  {why}: {root} -- no lib/libnvshmem_host.so*")
            continue
        version = _nvshmem_version(root)
        if version is None:
            rejected.append(f"  {why}: {root} -- cannot read include/non_abi/nvshmem_version.h")
            continue
        if version < _NVSHMEM_MIN:
            rejected.append(
                f"  {why}: {root} -- NVSHMEM {'.'.join(map(str, version))}, "
                f"need >= {'.'.join(map(str, _NVSHMEM_MIN))}"
            )
            continue
        print(f"-- NVSHMEM {'.'.join(map(str, version))} from {why}: {root}")
        return root, libs[-1]
    raise RuntimeError(
        "No usable NVSHMEM found. fast_ulysses needs NVSHMEM >= "
        f"{'.'.join(map(str, _NVSHMEM_MIN))} with include/nvshmem.h and "
        "lib/libnvshmem_host.so*. Candidates tried:\n" + "\n".join(rejected) + "\n"
        "Set NVSHMEM_HOME=<install root> to point at one explicitly."
    )


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        nvshmem_home, nvshmem_lib = _resolve_nvshmem()
        outdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        # Persistent build dir (not pip's ephemeral build_temp) so CMake can
        # build incrementally and skip recompiling unchanged TUs. CMake re-runs
        # configure automatically when the -D args change, so no manual wipe.
        builddir = _HERE / "build"
        builddir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        arch = env.get("FAST_ULYSSES_CUDA_ARCH", "80;90;100;120")
        # Torch expects dotted arch (e.g. 86 -> 8.6, 100 -> 10.0); insert the
        # decimal point before the last digit of each ";"-separated token.
        env["TORCH_CUDA_ARCH_LIST"] = " ".join(
            f"{tok[:-1]}.{tok[-1]}" for tok in arch.split(";") if tok
        )
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(_HERE),
                "-B",
                str(builddir),
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={outdir}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DEXT_SUFFIX={ext_suffix}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                f"-DNVSHMEM_HOME={nvshmem_home}",
                f"-DNVSHMEM_HOST_LIB={nvshmem_lib}",
            ]
            # Extra -D flags for odd setups (e.g. the CCCL::CCCL stub in docs/INSTALL.md).
            + shlex.split(env.get("FAST_ULYSSES_CMAKE_ARGS", "")),
            env=env,
        )
        # Cap -j at the CPU count (bare -j lets Make fork unbounded jobs, and nvcc
        # -t0 multiplies each TU further); CMAKE_BUILD_PARALLEL_LEVEL overrides.
        jobs = env.get("CMAKE_BUILD_PARALLEL_LEVEL") or str(os.cpu_count() or 1)
        subprocess.check_call(["cmake", "--build", str(builddir), f"-j{jobs}"], env=env)


setup(
    ext_modules=[CMakeExtension("fast_ulysses._C")],
    cmdclass={"build_ext": CMakeBuild},
)
