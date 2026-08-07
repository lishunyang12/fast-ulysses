# 安装

[English](../INSTALL.md) · [中文](INSTALL.md)

## 环境要求

- **PyTorch 2.10+**、Linux x86_64、CPython 3.10–3.13
- **CUDA 12.8+ 或 13**
- sm80 / sm90 / sm100 / sm120
- **NVSHMEM 3.4.5+** —— 来自 torch 本身就依赖的 `nvidia-nvshmem-cu1x` wheel，不需要单独安装。
  `NVSHMEM_HOME` 可以用一份自建的 NVSHMEM 覆盖它。
- 仅源码编译需要：CMake ≥ 3.18 和 `nvcc`。有 `ccache` 时会自动使用。

有两种组合没有 wheel，必须从源码编译：**torch ≤ 2.9**（它锁定 NVSHMEM 3.3.20，低于 3.4.5 这个下限）
和 **torch `+cu126`**（CUDA 12.6 无法生成 `sm_100`/`sm_120`）。

## 安装

```bash
pip install fast-ulysses
```

这个 wheel 是针对最新的稳定版 torch 编译的。其他受支持的 torch，请从 release 页面挑对应的 wheel ——
torch 的次版本号必须完全一致，CUDA 主版本号必须一致，tag 里的 CUDA 次版本号是一个下限：

| 你的 torch | torch 的 CUDA | wheel tag |
|---|---|---|
| 2.10.x | 12.x | `torch210cu128` |
| 2.10.x | 13.x | `torch210cu130` |
| 2.11.x | 12.x | `torch211cu128` |
| 2.11.x | 13.x | `torch211cu130` |
| 2.12.x | 12.x | `torch212cu129` |
| 2.12.x | 13.x | `torch212cu130` |
| 2.13.x | 12.x | `torch213cu129` |
| 2.13.x | 13.x | PyPI，见上 |

```bash
python -c "import sys,torch; print(torch.__version__, torch.version.cuda, sys.version_info[:2])"
pip install https://github.com/triple-mu/fast-ulysses/releases/download/v0.1.0/\
fast_ulysses-0.1.0+torch211cu128-cp311-cp311-manylinux_2_28_x86_64.whl
```

## 从源码编译

```bash
pip install -e . --no-build-isolation                              # all four architectures
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation    # one, much faster
```

| 变量 | 含义 |
| --- | --- |
| `FAST_ULYSSES_CUDA_ARCH` | 目标计算能力，用 `;` 分隔。默认 `80;90;100;120`。 |
| `NVSHMEM_HOME` | NVSHMEM 安装根目录，其中含 `include/nvshmem.h`。不设则用 torch 自带的那份。 |
| `CUDACXX` | CUDA 编译器；默认 `/usr/local/cuda/bin/nvcc`。 |
| `FAST_ULYSSES_BUILD_DIR` | CMake 构建目录。默认 `./build`，多次编译之间保留，因此重编是增量的。 |
| `FAST_ULYSSES_CMAKE_ARGS` | 透传给 CMake 的额外参数。 |

`--no-build-isolation` 是必需的：CMake 要通过已安装的 torch 定位 libtorch，所以编译时 torch 必须是
可导入的。

## NVLink fabric 损坏或缺失的节点

NVSHMEM 默认的 init 可能会去做 NVLS 多播映射或者拉起 IB remote transport，然后**段错误**。这个算子
只用单节点 P2P，两者都不需要，所以 `UlyssesGroup` 在构造时设置了安全的默认值
（`os.environ.setdefault` —— 要覆盖它们，必须在构造 group *之前*）：

```text
NVSHMEM_DISABLE_NVLS=1
NVSHMEM_REMOTE_TRANSPORT=none
```

NCCL 会自己去探测 NVLS，在 Fabric Manager 有问题的机器上会在 init 时死掉，报 `unhandled cuda
error` / “Failed to bind NVLink SHARP (NVLS) Multicast memory”。这种机器上要加 `NCCL_NVLS_ENABLE=0`。
这影响的是 `torch.distributed` 的 bootstrap 和 benchmark 里的参照实现，与我们无关。

## 导入失败时

`import fast_ulysses` 会捕获加载器的错误，并报出这个扩展是针对什么编译的、当前装的又是什么。
`fast-ulysses doctor` 会打印同样的信息，外加设备和 P2P 矩阵。三种常见原因：

- **`undefined symbol: _ZN3c10...`** —— wheel 是针对另一个 torch 次版本编译的。扩展继承了
  `c10d::Work` 并注册了一个 TorchScript class，这两样都扛不住次版本变化。按上面的表装对应你这个
  torch 的 wheel。
- **`libcudart.so.12: cannot open shared object file`** —— CUDA-12 的 wheel 装在了 CUDA-13 环境里，
  或者反过来。没有哪个 `LD_LIBRARY_PATH` 能解决这个问题；装正确的 wheel。
- **`libnvshmem_host.so.3: cannot open shared object file`** —— 缺 `nvidia-nvshmem-cu12`/`-cu13`。
  `pip install` 装上它，或者把 `LD_LIBRARY_PATH` 指向一份自建的 NVSHMEM。

## 其他编译问题

- **CMake `CMakeCache.txt directory ... is different than ...`** —— 保留下来的 `build/` 是从另一个
  路径配置的（仓库被移动过）。`rm -rf build` 然后重编。
- **`fatal error: cuda/std/array: No such file or directory`** —— CUDA 13 把 CCCL 的头文件挪到了
  `include/cccl/`。构建本身会加上这个路径；如果工具链目录结构不标准导致探测失败，就给两个编译器都
  加上，因为 host 端的编译单元也会 include `nvshmem.h`：

  ```bash
  CUDAFLAGS=-I/usr/local/cuda/include/cccl CXXFLAGS=-I/usr/local/cuda/include/cccl \
  pip install -e . --no-build-isolation
  ```
- **CMake `The link interface of target "nvshmem::nvshmem_host" contains: CCCL::CCCL`** —— CUDA-13 的
  NVSHMEM 配上了 CUDA-12 的工具链。这里 CCCL 只是头文件，所以一个 stub 就够了：

  ```bash
  printf 'if(NOT TARGET CCCL::CCCL)\n  add_library(CCCL::CCCL INTERFACE IMPORTED)\nendif()\n' > /tmp/cccl_stub.cmake
  FAST_ULYSSES_CMAKE_ARGS="-DCMAKE_PROJECT_INCLUDE=/tmp/cccl_stub.cmake" \
  pip install -e . --no-build-isolation
  ```
- **NVSHMEM 内部 init 段错误** —— 见上面的 fabric 一节，并确认每个 rank 都一起构造了
  `UlyssesGroup`。构造是集合操作。
