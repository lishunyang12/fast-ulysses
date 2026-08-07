# Installation

[English](INSTALL.md) · [中文](zh/INSTALL.md)

## Requirements

- **PyTorch 2.10+**, Linux x86_64, CPython 3.10–3.13
- **CUDA 12.8+ or 13**
- sm80 / sm90 / sm100 / sm120
- **NVSHMEM 3.4.5+** — comes from the `nvidia-nvshmem-cu1x` wheel torch already depends on;
  nothing to install separately. `NVSHMEM_HOME` overrides it with a site build.
- For a source build only: CMake ≥ 3.18 and `nvcc`. `ccache` is used when present.

Two combinations have no wheel and must be built from source: **torch ≤ 2.9** (it pins NVSHMEM
3.3.20, below the 3.4.5 minimum) and **torch `+cu126`** (CUDA 12.6 cannot emit `sm_100`/`sm_120`).

## Install

```bash
pip install fast-ulysses
```

That wheel is built against the newest stable torch. For any other supported torch, pick the
matching wheel from the release page — the torch minor must match exactly, the CUDA major must
match, and the CUDA minor in the tag is a floor:

| your torch | torch's CUDA | wheel tag |
|---|---|---|
| 2.10.x | 12.x | `torch210cu128` |
| 2.10.x | 13.x | `torch210cu130` |
| 2.11.x | 12.x | `torch211cu128` |
| 2.11.x | 13.x | `torch211cu130` |
| 2.12.x | 12.x | `torch212cu129` |
| 2.12.x | 13.x | `torch212cu130` |
| 2.13.x | 12.x | `torch213cu129` |
| 2.13.x | 13.x | PyPI, above |

```bash
python -c "import sys,torch; print(torch.__version__, torch.version.cuda, sys.version_info[:2])"
pip install https://github.com/triple-mu/fast-ulysses/releases/download/v0.1.0/\
fast_ulysses-0.1.0+torch211cu128-cp311-cp311-manylinux_2_28_x86_64.whl
```

## Build from source

```bash
pip install -e . --no-build-isolation                              # all four architectures
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation    # one, much faster
```

| Variable | Meaning |
| --- | --- |
| `FAST_ULYSSES_CUDA_ARCH` | Target compute capabilities, `;`-separated. Default `80;90;100;120`. |
| `NVSHMEM_HOME` | NVSHMEM install root containing `include/nvshmem.h`. Omit to use torch's copy. |
| `CUDACXX` | CUDA compiler; defaults to `/usr/local/cuda/bin/nvcc`. |
| `FAST_ULYSSES_BUILD_DIR` | CMake build tree. Default `./build`, kept between builds so rebuilds are incremental. |
| `FAST_ULYSSES_CMAKE_ARGS` | Extra flags passed through to CMake. |

`--no-build-isolation` is required: CMake locates libtorch through the installed torch, so torch
has to be importable at build time.

## Nodes with a broken or absent NVLink fabric

NVSHMEM's default init may attempt the NVLS multicast mapping or the IB remote transport and
**segfault**. This operator is single-node P2P and uses neither, so `UlyssesGroup` sets safe
defaults at construction (`os.environ.setdefault` — override them *before* constructing the group):

```text
NVSHMEM_DISABLE_NVLS=1
NVSHMEM_REMOTE_TRANSPORT=none
```

NCCL probes NVLS on its own and, on a broken Fabric Manager, dies at init with `unhandled cuda
error` / "Failed to bind NVLink SHARP (NVLS) Multicast memory". Run with `NCCL_NVLS_ENABLE=0`
there. This affects the `torch.distributed` bootstrap and the benchmark references, not us.

## When the import fails

`import fast_ulysses` catches the loader error and reports what the extension was built against
next to what is installed. Run `fast-ulysses doctor` for the same block plus the device and P2P
matrix. The three common causes:

- **`undefined symbol: _ZN3c10...`** — the wheel was built for a different torch minor. The
  extension subclasses `c10d::Work` and registers a TorchScript class, neither of which survives a
  minor bump. Install the wheel for your torch, from the table above.
- **`libcudart.so.12: cannot open shared object file`** — a CUDA-12 wheel in a CUDA-13 environment
  or the reverse. No `LD_LIBRARY_PATH` fixes this; install the right wheel.
- **`libnvshmem_host.so.3: cannot open shared object file`** — `nvidia-nvshmem-cu12`/`-cu13` is
  missing. `pip install` it, or point `LD_LIBRARY_PATH` at a site NVSHMEM.

## Other build problems

- **CMake `CMakeCache.txt directory ... is different than ...`** — the persistent `build/` was
  configured from another path (the repo moved). `rm -rf build` and rebuild.
- **`fatal error: cuda/std/array: No such file or directory`** — CUDA 13 moved the CCCL headers
  into `include/cccl/`. The build adds that path itself; if a non-standard toolkit layout defeats
  the probe, add it to both compilers, since the host translation units include `nvshmem.h` too:

  ```bash
  CUDAFLAGS=-I/usr/local/cuda/include/cccl CXXFLAGS=-I/usr/local/cuda/include/cccl \
  pip install -e . --no-build-isolation
  ```
- **CMake `The link interface of target "nvshmem::nvshmem_host" contains: CCCL::CCCL`** — a
  CUDA-13 NVSHMEM against a CUDA-12 toolkit. CCCL is header-only here, so a stub satisfies it:

  ```bash
  printf 'if(NOT TARGET CCCL::CCCL)\n  add_library(CCCL::CCCL INTERFACE IMPORTED)\nendif()\n' > /tmp/cccl_stub.cmake
  FAST_ULYSSES_CMAKE_ARGS="-DCMAKE_PROJECT_INCLUDE=/tmp/cccl_stub.cmake" \
  pip install -e . --no-build-isolation
  ```
- **Init segfault inside NVSHMEM** — see the fabric section above, and check that every rank
  constructs `UlyssesGroup` together. Construction is collective.
