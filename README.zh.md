<div align="center">

# fast-ulysses

**把 Ulysses 序列并行的 all-to-all 做成 torch 自定义算子,数据由 GPU 复制引擎经 NVSHMEM 对称内存搬运。**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/triple-mu/fast-ulysses/blob/master/LICENSE)

[English](https://github.com/triple-mu/fast-ulysses/blob/master/README.md) · [中文](https://github.com/triple-mu/fast-ulysses/blob/master/README.zh.md)

</div>

Ulysses 序列并行把长序列切到多张卡上:注意力之前用一次 all-to-all 把序列分片换成头分片,之后
再换回来。对长序列视频 DiT 这类负载,这两次集合通信就是关键通信。

本仓库把这个 4D all-to-all 做成 `torch.ops.fast_ulysses.all_to_all_single_4d`。传输是带 pitch 的
`cudaMemcpy2D/3DAsync`,直接写进对端对称堆的地址,因此**不占用任何 SM**,可以在计算核心占满所有
SM 槽位时并行跑;序列/头的重排则表达成拷贝的 stride,不需要两个独立的 permute kernel。

## 做了什么

- **零 SM 传输**。发往远端的拷贝串行在一条流上,本 rank 自己那份走调用者的流。
- **重排不额外收费**。寻址是一份 host 侧的 plan(`a2a_plan.cpp`),没有 GPU 也能测,所以没有
  launch 配置、没有 autotune。
- **默认拷贝,借用要写明**。`all_to_all_single_4d` 返回调用者拥有的张量;
  `all_to_all_single_4d_borrowed` 直接返回对称窗口本身 —— 少一次拷贝,但只在同一 tag 的下一次
  调用之前有效。
- **异步**。返回 `AsyncCollectiveTensor`,对它做的第一个 aten 算子会自己等待。`barrier=False`
  可以让若干次借用调用(一层的 q/k/v)共用一次收尾握手。
- **不等长分片**。按 rank 给 `seq_splits` / `head_splits`,这正是调用方能去掉序列 padding 的原因。
  等长是特例,不是另一条路径。
- **先 reserve 再 seal**。`reserve()` 预先定好进程会用到的每个窗口;之后未声明的调用会直接报错,
  而不是在集合通信中间做一次分配。

限制:单节点,`world_size ∈ [1, 8]`(含奇数);`float16` / `bfloat16`;`d * elem_size` 按 16 字节
对齐;组内每一对 GPU 必须 P2P 可映射(`fast-ulysses doctor` 会打印矩阵,不可达的一对在构造时就被
拒绝)。

融合示例(把 QK RMSNorm + RoPE 融进 scatter kernel)在 `examples/qk-norm-rope-fusion` 分支。

## 效果

8 卡,每一行一个独占分配的节点,同一个容器、同一个 `.so`(内含四种架构的 SASS)。`base` 是
`torch.distributed` 的 permute + `all_to_all_single` + permute;`raw` 是裸的
`all_to_all_single`,不含让结果可用的那次重排。Wan 720p,bf16,单位 ms。

| GPU | 互联 | base | 我们 | vs base | transfer | vs raw |
|---|---|---|---|---|---|---|
| A100-SXM4-80GB | NVLink | 2.865 | **1.670** | **1.72×** | 1.227 | 1.12× |
| H200 | NVSwitch | 1.575 | **0.855** | **1.84×** | 0.683 | 1.22× |
| B200 | NVLink | 1.193 | **0.554** | **2.15×** | 0.402 | 1.22× |

三代硬件上形状一致:baseline 里 47–60% 是重排,而这部分对我们不要钱;单看传输也比裸的
`all_to_all_single` 快 1.12–1.37×。在并发的 GEMM 链下,这次集合通信基本被完全藏住(B200 86%,
A100 约 105%)。去掉序列 padding 对我们是免费的(1.00×),baseline 为同样的改动要付 5–8%。

完整的分阶段表格、五台机器:[docs/zh/BENCHMARK.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/zh/BENCHMARK.md)。

## 什么情况下我们不占优

**PCIe 机器上跨两个 CPU socket 的组。** 组在一个 socket 内时是 1.4–2.2×,和 NVLink 上一样;跨
socket 时约为 `torch.distributed` 的 0.62×。原因不是我们的传输慢,而是 `all_to_all_single`
在那里根本不用 GPU 直连 P2P —— 它绕开 socket 边界,走 InfiniBand 网卡或 host 共享内存,而我们
一律直接写对端显存。把这条旁路关掉,同一条路径上我们快 3.8–4.9 倍。`fast-ulysses doctor` 会在
组跨 socket 时报出来;测量数据见
[docs/zh/BENCHMARK.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/zh/BENCHMARK.md#two-socket-pcie)。

如果两种机器都要跑,用 `make_group` 让它自己选:

```python
from fast_ulysses import make_group

group = make_group(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)
# 当它因为 GPU 跨 socket 而选了 torch.distributed 时,group.fallback 为 True
```

跨 socket 时返回 `TorchUlyssesGroup`(同样四个入口,底下是 `torch.distributed`),否则返回
`UlyssesGroup`。`prefer="fast"` / `prefer="torch"` 可以强制。两者在每个入口、等长和不等长两种
形状上都**逐位一致**,所以调用方只需要一套代码;区别只是回退这条路没有重叠收益,返回值也没有
生命周期约束。

## 安装

需要 **PyTorch 2.10+**、**CUDA 12.8+ 或 13**,以及 sm80 / sm90 / sm100 / sm120 的卡。NVSHMEM
3.4.5+ 来自 torch 本身就依赖的 `nvidia-nvshmem-cu1x` wheel,不需要单独装。

```bash
pip install fast-ulysses                                  # 最新 torch,从 PyPI
pip install -e . --no-build-isolation                     # 源码编译,四种架构
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation   # 单一架构,快得多
```

其他 torch 版本的 wheel,以及导入失败时怎么查:[docs/zh/INSTALL.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/zh/INSTALL.md)。

## 最小示例

`torchrun --nproc_per_node=2 example.py`:

```python
import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup

dist.init_process_group("nccl")
rank, ws = dist.get_rank(), dist.get_world_size()
lr = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(lr)

group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)

# mode 0: (b, s_local, n_global, d) -> (b, s_global, n_local, d)
b, s_local, d = 2, 16, 128
x = torch.randn(b, s_local, 4 * ws, d, dtype=torch.bfloat16, device=f"cuda:{lr}")

# 每个 rank 必须发出完全相同的 (shape, mode, tag) 调用序列。
out = group.all_to_all_single_4d(x, mode=0, tag="demo")
assert out.shape == (b, s_local * ws, 4, d)

group.destroy()
dist.destroy_process_group()
```

## 接口

| 接口 | 说明 |
| --- | --- |
| `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)` | 集合操作:NVSHMEM 初始化 + 对称堆内存池。 |
| `group.reserve(calls, *, allow_growth=False)` | 预先定好每个窗口,然后封死。 |
| `group.all_to_all_single_4d(x, *, mode=0, tag="", out=None)` | 默认入口,返回调用者拥有的张量。 |
| `group.all_to_all_single_4d_borrowed(x, *, mode=0, tag="")` | 不做 copy-out,返回的就是窗口本身。 |
| `group.all_to_all_single_4d_async(...)` | 默认入口,跑在高优先级通信流上。 |
| `group.all_to_all_single_4d_borrowed_async(..., barrier=True)` | 借用入口,跑在同一条流上。 |
| `group.destroy()` | 释放对称堆资源(集合操作)。 |
| `fast-ulysses doctor` | 构建信息、设备、P2P 矩阵、socket 分布。 |

代码为什么长这样,以及它依赖了哪些没有文档保证的行为:[docs/zh/DESIGN.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/zh/DESIGN.md)。

形状约定、tag 语义、barrier 的顺序契约,以及**集合操作的硬约束**(违反 rank 一致的调用序列会让
整个组挂死):[docs/zh/API.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/zh/API.md)。

## 测试

```bash
pytest                                                            # 少于 2 卡自动跳过
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py  # 直接跑单个 worker
```

开发环境配置:[docs/zh/DEVELOP.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/zh/DEVELOP.md)。

## 许可证

Apache-2.0,见 [LICENSE](https://github.com/triple-mu/fast-ulysses/blob/master/LICENSE)。
