# 开发

[English](../DEVELOP.md) · [中文](DEVELOP.md)

## 环境配置

```bash
FAST_ULYSSES_CUDA_ARCH=<your arch, e.g. 90> pip install -e ".[dev]" --no-build-isolation
pre-commit install
```

构建会复用保留下来的 `build/` 目录，所以改一次编一次只会重新编译改动过的编译单元（装了 `ccache`
的话还会更快）。`NVSHMEM_HOME` 是可选的 —— 见 [INSTALL.md](INSTALL.md)。

## 代码检查

pre-commit 是唯一的入口：

```bash
pre-commit run --all-files
```

Python 用 **ruff**（check + format，行宽 100，py310+）。`fast_ulysses/csrc/` 下的 C++/CUDA 用
**clang-format**，在 `.pre-commit-config.yaml` 里锁定为 **v15.0.7** —— 代码就是用这个版本格式化的。
锁定的版本和本地的二进制要保持一致。

## 测试

```bash
pytest                      # everything runnable here
pytest -m "not multigpu"    # host-only, no GPU needed
pytest -m multigpu          # the torchrun-wrapped suites
```

`tests/test_plan.py` 在 numpy buffer 上重放寻址逻辑（`csrc/a2a_plan.cpp`），与
`all_to_all_single` + permute 的参照实现对比。它不需要 GPU，也不需要 process group，只需要编译好的
扩展 —— 这也是 CI 唯一能跑的正确性检查。

`tests/test_multigpu.py` 把 `tests/distributed/` 下的每个 worker 作为 `torch.distributed.run` 子进程
拉起来，少于 2 张卡时跳过。`FAST_ULYSSES_TEST_NPROC` 可以覆盖进程数（`=3` 用来测奇数 world size）。

Worker 保持可以直接运行，方便调试：

```bash
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py
```

| worker | 断言了什么 |
|---|---|
| `a2a_correctness` | 与 torch 的 permute + a2a + permute 参照实现逐位相等 |
| `a2a_fallback` | `TorchUlyssesGroup` 与 `UlyssesGroup` 逐位相等,且 `make_group` 的选择跟随 socket 布局 |
| `a2a_async` | 异步结果与同步结果一致，并且重叠窗口是真实存在的 |
| `a2a_uneven` | `seq_splits` / `head_splits` 与 `dist.all_to_all` 在不等长张量上的对比 |
| `a2a_copy_out` | 拷贝形式拥有自己的结果，借用形式不拥有 |
| `a2a_subgroup`、`a2a_subgroup_divergent` | 两个 stride-2 的子组同时存在，形状相同和不同两种情况 |
| `a2a_torch_nvshmem_coexist` | 这个扩展和 torch 自己的 NVSHMEM 共存于一个进程 |

下面六个是**对抗性**的：每个都构造一种特定的不安全时序，只断言结果没有被撕裂。

| worker | 它构造的时序 |
|---|---|
| `a2a_window_race` | 我们还在读自己的窗口时，对端的下一次调用已经到了 |
| `a2a_cudagraph` | 被 capture 的调用重放一次，对照设备侧的 epoch |
| `a2a_ce_flag_ordering` | 宣告 payload 的 flag 到达时，payload 本身是否可见 |
| `a2a_ce_fault_injection` | 用 `_set_ce_fault` 按需打破那个顺序 |
| `a2a_overlapping_barriers` | 一个 tag 上的异步调用对上另一个 tag 上的同步调用 |
| `a2a_alias_guard` | 输入或 `out` 与该 tag 自己的窗口重叠 |

一个对抗性 worker 的价值，恰好等于它构造的那个时序的价值，而时序会失效：改一次 barrier，或者换一台
更快的机器，就可能让各 rank 重新对齐，留下一个通过了但什么也没测的 worker。所以每个 worker 都在模块
docstring 里点名自己的**负对照** —— 删掉哪一行会让它失败，以及失败长什么样。**任何 barrier 改动之后
都要重跑这些负对照**；加上负对照仍然通过，说明这个 worker 是瞎的，而不是说明代码是安全的。
`a2a_ce_fault_injection` 是例外：它每次运行都自己上负对照。`a2a_cudagraph` 会打印
`captured=True/False` —— 一次 `captured=False` 的绿色运行什么也没检查。

`a2a_overlapping_groups.py` 没有被注册，也不能注册：它构造的两个组是互相重叠而不是划分的，按设计就会
挂死。它记录的是一条约束，不是一个测试。

## 发版

CI 没有 GPU runner，所以 `tests/distributed/` 下的东西在那里永远不会跑。CI 能证明的是：每个配置都能
编出四种架构、链接的库正好是预期的那些且 RUNPATH 可重定位、能在目标 torch 下加载，并且通过
`test_plan.py`。

```bash
scripts/build_wheels.sh          # one (torch, CUDA) row, inside a manylinux builder
scripts/check_wheel.py <whl>     # the ELF/metadata gate; also runs inside build_wheels.sh
scripts/preflight_gpu.sh <whl>   # MANDATORY before a tag: the built wheel on a real multi-GPU box
```

`preflight_gpu.sh` 会打印一段可以直接放进 release notes 的信息。至少要为最新的 torch 那一行和一个
CUDA-12 的行跑一遍；最老的那几行只有“能编译、能加载”这一层证据，release notes 里应当写明这一点。

Benchmark 必须在 `scripts/exclusive.sh` 下运行，它会一直拒绝启动直到请求的 GPU 空闲，并打印
`EXCLUSIVE` 或 `CONTENDED`。`CONTENDED` 的数字不是数字。

## 目录结构

```
fast_ulysses/          Python 包（comm.py: UlyssesGroup；cli.py: doctor）
fast_ulysses/csrc/     C++/CUDA 源码（bindings.cpp 注册 torch library）
tests/                 pytest 用例；tests/distributed/ 放 torchrun worker
benchmark/             分阶段拆解、padding 开销、GEMM 重叠，以及一个 nsys/ncu 驱动脚本
scripts/               GPU 独占包装、wheel 构建与校验、发版前的 preflight
docs/                  本文档；docs/zh/ 是中文翻译
```
