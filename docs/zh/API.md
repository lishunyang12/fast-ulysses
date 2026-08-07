# API 参考

[English](../API.md) · [中文](API.md)

所有东西都从顶层包导出：`from fast_ulysses import UlyssesGroup, CompletedHandle`。
形状记号：`b` batch，`d` head dim，`ws = world_size`；`s_local` / `n_local` 是**本 rank 的**序列分片和
头分片，`s_global = sum(seq_splits)`，`n_global = sum(head_splits)`。不等长分片是一般情况 ——
`s_local = s_global / ws` 只是等长的特例。

## 集合操作的硬约束

**违反其中任何一条都会让整个组挂死。** 不会报错，也不会超时。

- **每个 rank 必须发出完全相同的 `(shape, mode, tag)` 序列**，同步与异步、拷贝与借用的调用都算数
  —— 只有一个序列，不是两个。不一致会让对称堆的窗口（新的 `tag`+容量+dtype 是集合分配的）和
  barrier 一起分叉。
- **同一个 tag 内的调用必须保持有序**，因为 epoch 协议要求每个 rank 对一个 tag 的握手编号完全一致：
  在该 tag 的下一次调用之前，先等待尚未完成的异步结果。**不同 tag 之间则不必** —— 两个 tag 可以在互不
  排序的 stream 上同时在飞。
- **构造、`reserve()` 和 `destroy()` 是对整个 job 的集合操作**，不是对 `process_group` 的；在 2-D 并行
  下，这意味着两个 sp 组必须一起来。
- **`barrier=False` 的模式在每个 rank 上必须完全一致。**

## `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)`

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `process_group` | `ProcessGroup` 或 `None` | 用于 bootstrap 的 process group；`None` 表示 `dist.group.WORLD`。子组也可以，只要它的 rank 是**等间隔**的（它们会变成一个 NVSHMEM strided team）—— 例如 tp2 × sp4 网格里的 sp 切片 `{0,2,4,6}`。不构成等差数列的 rank 列表会报错。 |
| `device` | `torch.device` 或 `None` | 本 rank 的 CUDA 设备；`None` 表示当前设备。 |
| `initial_pool_bytes` | `int` | 内存池，默认 `2<<30`（2 GiB）。构造时由一次对称分配**整块**拿走 —— 是实打实占用，不是上限 —— 之后每次集合通信的窗口都是它里面的一个偏移。堆的大小由**第一个存活的**组决定；之后更大的请求只会告警，而销毁所有组之后下一个组可以重新定尺寸。 |

组内每一对 rank 都必须 P2P 可映射（`nvshmem_ptr` 非空，这跟随 `cudaDeviceCanAccessPeer`）—— NVLink
和 PCIe 都算，包括双 socket 机器。不可达的一对会在构造时被拒绝并点名，`fast-ulysses doctor` 可以提前
打印整张矩阵。

## `reserve(calls, *, allow_growth=False) -> None`

预先定好本进程会用到的每一个对称窗口，然后封死内存池。每一条是一个 mapping，含 `tag`、`shape`（4D 的
**输入**形状），以及可选的 `mode`（0）、`dtype`（`bfloat16`）、`seq_splits`、`head_splits`。窗口按容量
匹配：给每个 tag 它会见到的最大形状即可。

```python
group.reserve([{"tag": "qkv", "shape": (b, s_local, n_global, d), "mode": 0},
               {"tag": "qkv", "shape": (b, s_global, n_local, d), "mode": 1}])
```

封死之后，**未声明的调用会报错**而不是去分配，于是形状往上漂就是一个错误，而不是每涨一次留下一个废弃
的窗口；`allow_growth=True` 会跳过封死。每个 rank 的条目和顺序都要相同 —— 在那之后，只要每次调用都装
得进某个已声明的容量，各组就可以自由发散。

## 形状、分片与 tag

| mode | 输入 `x` | 输出 |
| --- | --- | --- |
| 0 —— 散开 head、聚拢序列 | `(b, s_local, n_global, d)` | `(b, s_global, n_local, d)` |
| 1 —— 它的逆 | `(b, s_global, n_local, d)` | `(b, s_local, n_global, d)` |

`seq_splits[p]` 是 rank p 的序列长度，`head_splits[p]` 是它的头数。**要么都传，要么都不传**，**每个
rank 上完全一致**，并且和实际交进来的形状对得上。都不传就是等长分片，此时被散开的那一维必须整除
（mode 0 是 `n_global % ws`，mode 1 是 `s_global % ws`）；都传则允许分片任意不等长，这正是调用方能去掉
序列 padding 的原因。窗口按**最大**那个 rank 的输出来分配（分配是集合操作），每个结果都是它的一段密集
前缀。

一个 `tag` 命名一个对称窗口（按 `tag`+容量+dtype，保持在历史最高容量）以及与之配套的握手状态 ——
flag buffer 和 epoch 计数器，按 tag、在设备上（`csrc/ulysses_group.cuh` 里的 `BarrierState`）。窗口从该
tag 的第一次调用一直活到 `destroy()`，而该 tag 上的每一次调用都会覆盖它，所以在借用形式下同时存活的
结果需要不同的 tag；拷贝形式则不需要。

## 会抛什么

`RuntimeError`，来自在**该次调用的第一个 barrier 之前**就跑完的校验，因此被拒绝的参数不会留下任何
rank 在等那些没有拒绝它的对端：`x` 不是 4D 或不在 CUDA 上，dtype 不是 `float16`/`bfloat16`，
`d * elem_size` 不是 16 B 对齐，`mode` 不是 0 或 1，`world_size` 不在 `[1, 8]` 内；`seq_splits` /
`head_splits` 只传了一个，或者分片与 `x` 的形状矛盾；没传分片而被散开的那一维不能整除；`out` 不是连续
的 CUDA 张量，或者它的 dtype 或形状不是输出的；`x` 或 `out` 与该 tag 的窗口重叠；封死之后调用超出了已
声明的容量。

## `all_to_all_single_4d(x, *, mode=0, tag="", out=None, seq_splits=None, head_splits=None) -> Tensor`

**默认入口。** 返回一个调用者拥有的张量，不附带任何生命周期规则：它可以活过该 tag 的下一次调用、在另
一条 stream 上读、交还给 allocator，或者在 `destroy()` 之后继续存在。`x` 是 4D CUDA
`float16`/`bfloat16`，内部会调 `.contiguous()`；`out` 是可选的预分配目的地，按上面的规则校验，`None`
则新分配。

传输落进该 tag 的窗口，然后一次平坦的 device-to-device 拷贝在调用者的流上、收尾握手之后把它搬出来 ——
这次拷贝就是代价，它作为 `all_to_all_single_4d_timed` 的 `copy_out` 阶段被报出来。传输只有一条路径，
永远如此：带 pitch 的 `cudaMemcpy2D/3DAsync` 直接写进对端窗口的地址，寻址来自 host 侧的 plan
（`csrc/a2a_plan.cpp`），没有 launch 配置也没有 autotune，因此第一次调用天然就是集合安全的。数据见
[docs/zh/BENCHMARK.md](BENCHMARK.md)。

## `all_to_all_single_4d_borrowed(x, *, mode=0, tag="", seq_splits=None, head_splits=None) -> Tensor`

同一个集合通信，只是没有 copy-out：**返回的结果就是该 tag 的对称窗口本身。** 形状、分片和集合约束都和
上面一样；没有 `out`，也没有 `barrier` —— 一个被推迟的同步结果会是一个没人来发布、因而读不了的 view。
代价是一份**本库不做任何强制的生命周期契约** —— 没有检查、没有断言、没有 debug 模式：

- **只在该 tag 的下一次调用之前有效**。那次调用的传输会写同样的字节，这个结果会不声不响地变成那一个。
- **在产生它的那条 stream 上消费它**，并且要在那次调用之前。要在另一条 stream 上读，同步是你自己的事。
- **不要在 `destroy()` 之后再读它** —— 内存已经释放。
- `.clone()`，或者任何产生新张量的算子，是留住它的办法。

跨 rank 的安全**是**处理好的：在每个 rank 都到达下一次调用的开场 barrier 之前，没有对端能覆盖这个
窗口，而你的读在你这条 stream 上被排在那个 barrier 之前
（`tests/distributed/a2a_window_race.py`、`a2a_copy_out.py`）。拿不准就用拷贝形式。

## `all_to_all_single_4d_async(x, *, mode=0, tag="", out=None, seq_splits=None, head_splits=None)`

把拷贝形式的集合通信提交到组里那条高优先级的通信流上并立即返回，把调用者拥有的张量包进一个
`AsyncCollectiveTensor`；参数和集合契约与同步调用相同。`result.wait()` 返回原始张量，**任何 aten 算子
对结果的第一次使用**也一样 —— 两种方式都是调用者的当前流在 GPU 侧等待通信流的完成事件，host 不阻塞。
**view 算子**不等待，它只是重新包一层。

**每一个结果都要等，或者都要用。** 被直接丢掉的结果会把它在 torch work registry 里的条目和它的 CUDA
event 留下来，torch 会在进程退出时打印幸存者的数量。`out=` 是唯一的漏洞 —— 直接读你自己的 `out` 根本
不会碰到 registry，所以要读返回的那个 wrapper。如果 `libtorch_cpu` 里没有 `c10d::register_work`，就没有
registry 可以挂靠，两种异步形式都会返回 `CompletedHandle`：同样的 `.wait()`、正确的结果、没有重叠。
类型不同，所以这件事是看得见的。

**两种异步形式**都会在调用者的流上把输入拷进一个常驻的、按 `(tag, shape, dtype)` 索引的 buffer，只有
通信流去读它，因此 `x` 永远不会被跨流持有；代价是每次调用一次设备拷贝，以及 `tags × 张量大小` 的常驻
显存。与同步调用混用：只能用不同的 tag。

## `all_to_all_single_4d_borrowed_async(x, *, mode=0, tag="", barrier=True, seq_splits=None, head_splits=None)`

一个覆盖在**窗口 view** 上的 `AsyncCollectiveTensor`，遵守上面那份生命周期规则，其中“产生它的那条
stream”从等待之后起就是调用者的流。等待绑定的是那**次调用**，不是那个窗口 —— 每个借用结果都是一个带
自己 storage 的新 view，而 registry 正是按它来做 key —— 所以等待本身完全不能说明该 tag 后来的调用是否
已经把字节覆盖掉了。

**成组握手（`barrier=False`）**：每次调用都以一次握手**开场**（写者等读者），这一次不是可选的；
`barrier=False` 只推迟**收尾**的那一次，推给之后**同一条 stream 上**某次 `barrier=True` 的调用，于是
若干次异步调用 —— 一层的 q、k、v，tag 各不相同 —— 共用一次收尾握手，把每个组的 2N 次减掉 N-1 次。
发布是按 stream 顺序而不是按 tag，所以在带 barrier 的那个结果被等到之前，`barrier=False` 结果的 view
都不能安全地读。**所有 rank 必须使用完全一致的模式。** 这个 flag 只存在于借用形式上：拷贝形式会在对端
的写落地之前就把窗口拷出去。

## `destroy() -> None`

释放对称堆的资源（排空通信流、`dist.barrier`、销毁）；所有 rank 必须一起调用。丢掉一个组而不调用它，
会带着一条告警泄漏整个堆：拆除是集合操作。

## `UlyssesGroup` 上的其他入口

| 入口 | 用途 |
| --- | --- |
| `all_to_all_single_4d_timed(x, *, mode=0, tag="", seq_splits=None, head_splits=None)` | 拷贝形式的调用，返回 `(output, {barrier_in, transfer, barrier_out, copy_out} in ms)`。**仅用于 benchmark**：读取这些 event 会同步设备。 |
| `barrier_epoch(tag) -> int` | 该 tag 在设备侧的握手计数器，在该 tag 的第一次调用之前是 0。**仅用于测试**：读它会同步设备。 |

## `make_group(process_group=None, device=None, initial_pool_bytes=2<<30, prefer="auto")`

返回在这台机器上更快的那个组类。和两个构造函数一样,是集合操作。

- `prefer="auto"`(默认)—— 组内 GPU 跨多个 CPU socket 时返回 `TorchUlyssesGroup`,否则返回
  `UlyssesGroup`。无法判定 socket 布局时按"不跨"处理,因为那就是单 socket 的情形。
- `prefer="fast"` / `prefer="torch"` —— 强制其中一个。

`result.fallback` 对 `TorchUlyssesGroup` 是 `True`,对 `UlyssesGroup` 是 `False`。

`spans_sockets(process_group=None) -> bool | None` 是单独的同一个检查。它是集合操作:每个 rank
只能读到自己那张卡的 NUMA node,所以要汇总。`None` 表示至少有一张卡的 node 内核没有报出来。

## `TorchUlyssesGroup(process_group=None, device=None, initial_pool_bytes=...)`

上面那四个集合操作的 `torch.distributed` 实现,用于唯一那种它更快的拓扑。在每个入口、两种形状
族上与 `UlyssesGroup` **逐位一致**。

放宽的部分:返回值永远是调用者拥有的,所以借用形式没有生命周期约束;`tag` 被忽略,因为调用之间
不复用任何东西;异步形式在返回前就已完成,所以没有重叠收益;`reserve()` 和 `destroy()` 是空操作。
集合调用序列的契约仍然适用,因为 `torch.distributed` 自己就有。

# 环境变量

`UlyssesGroup.__init__` 会在 NVSHMEM init 之前设置 `NVSHMEM_SYMMETRIC_SIZE`，并以 `setdefault` 的方式
设置 `NVSHMEM_DISABLE_NVLS` / `NVSHMEM_REMOTE_TRANSPORT`。这些以及构建相关的变量都记录在
[docs/zh/INSTALL.md](INSTALL.md)。
