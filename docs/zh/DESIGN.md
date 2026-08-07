# 设计说明

[English](../DESIGN.md) · [中文](DESIGN.md)

代码为什么长成这样,以及它依赖了哪些没有文档保证的行为。接口契约见 [API.md](API.md),数字见
[BENCHMARK.md](BENCHMARK.md)。

## 传输

对端窗口用普通的 `cudaMemcpy2D/3DAsync` 写入,地址来自 `nvshmem_ptr`。这条路走的是复制引擎,
**不占用任何 SM** —— 这正是要点:一个驻留在 SM 上的集合通信在 GEMM 占满所有 SM 时拿不到 block
槽位,而这个不需要。

序列/头的重排表达成这些拷贝的源/目的 stride,所以除了本来就必须发生的传输之外不额外花钱。这就是
为什么 baseline 的两个 permute kernel 在我们这边根本不出现。

全部寻址逻辑在 `csrc/a2a_plan.cpp` 里,不含 CUDA、不含 NVSHMEM。因此布局契约可以在没有 GPU 的
机器上测(`tests/test_plan.py`),这也是 CI 唯一能跑的正确性检查。

不等长分片是通例,等长只是 `seq_splits = [s/P] * P`。只有一条代码路径,所以只有一处需要做对。

## 只做一次对称内存分配

内存池在构造函数里用一次 `nvshmem_align` 拿下整块对称堆,`acquire()` 此后只发放偏移量。

这不是为了整洁。`nvshmem_align` 是集合操作,内部会同步 CUDA stream,而 `barrier=False` 是**故意**
把一个自旋 barrier 留在飞行中的。因此在调用路径上做分配,会把 host 停在 `nvshmem_align` 里 ——
在那里它再也发不出对端正在自旋等待的那次发布,而对端的 host 也停在同一个地方。循环等待。

各 rank 的本地偏移之所以能对齐,只因为每个 rank 都按同样的顺序发放,而这本来就是 SPMD 调用契约的
要求。`reserve()` + `seal()` 的作用,是把违反这个顺序变成一个错误,而不是变成一个 rank 去寻址另一个
rank 的窗口。

## barrier

一个单 block 的自旋 kernel,用 release store 发布、用 acquire load 等待,作用在一个驻留在设备上的
epoch 计数器上。epoch 放在设备上而不是在 host 上算,是为了让 CUDA graph 的捕获能正确重放 —— host
上算出来的 epoch 会把一个常量烘焙进 graph。

`cuStreamWriteValue64` / `cuStreamWaitValue64` 可以把最后一次 kernel 启动也从路径上去掉,试过。
两条不利证据:在并发计算下它们测出来**更差**,而避免这种回退正是这个算子存在的理由;而且等待那
一侧需要一个 remote-write-flush 设备属性,目标硬件里有相当一部分没有。自旋 kernel 的内联 PTX 只
需要 `sm_70`。

## 它依赖了什么没有文档保证的东西

**一次已完成的复制引擎写入,在稍后某个 kernel 的 release store 到达时,在目的端是可见的。**
没有任何厂商文档这么说:

- CUDA API reference 把 memcpy 的完成定义为一个 **host 侧**属性;
- Programming Guide 的跨设备顺序保证只作用于 NULL stream,并且对非默认 stream 上的异步拷贝被撤销;
- PTX 把 `.release` 的作用域限定为"当前线程的先前操作",而复制引擎的传输不是;
- NVSHMEM 和 NCCL 都没有把 host 发起的 CE 传输与 SM 的 release store 配对使用,两者都把 flag 放在
  数据自己的路径上。

它在测试中成立,这是证据,不是保证。`tests/distributed/a2a_ce_flag_ordering.py` 测它,
`tests/distributed/a2a_ce_fault_injection.py` 是让那个测试保持诚实的反例控制 —— 它在每次运行时都
把故障武装一遍,所以那个测试不可能在无声中失去测试能力。

## 两个 NVSHMEM 入口用的不是文档里那个

用 `nvshmemx_hostlib_init_attr` 而不是内联的 `nvshmemx_init_attr`,用 `nvshmemx_hostlib_finalize`
而不是 `nvshmem_finalize`。内联版本会调用 `nvshmemi_init_thread` / `nvshmemi_finalize`,这两个符号
只存在于静态库 `libnvshmem_device.a` 里。链接它会和 torch 自带 NVSHMEM 的版本节点冲突,表现为
`undefined symbol: nvshmem_selected_device_transport`。`hostlib_` 这一组是 host 共享库直接导出的
入口,NVSHMEM 自己的 Python unique-id 路径也用它们。

这同时也是为什么只链接 host 库、并且关掉 `CUDA_SEPARABLE_COMPILATION`:这些 kernel 里没有任何
设备侧的 `nvshmem_*` 调用,所以没有东西需要设备库。

## 写明了但没有强制的约束

**同时存活的组必须构成对整个作业的划分。** 有两次强制它的尝试被写出来又被删掉了,而不是留在那里
装作是一层保护:

1. 对已建立的 PE 集合做**本地**检查看不见这个违规 —— 分歧在**第一次**构造时就已经存在了,那时还
   没有第二个组。
2. 对 PE 集合做 **all-gather** 也不行。gather 本身是对整个 world 的集合操作,而只有组成员才会走到
   构造函数;不加入任何组的 rank 永远不会到达,于是 gather 就在它本该保护的那次 split 的位置挂死。

能成立的检查必须是**每个** rank 都会调用的,所以它不能待在只有成员才调用的构造函数里。没有实现。
`tests/distributed/a2a_overlapping_groups.py` 演示了这个失败,并且**故意**没有注册,因为跑它会挂。

**借用形式的结果只在同一 tag 的下一次调用之前有效。** 没有任何东西强制这一点。真正被强制的是更窄
的那种情况:输入或 `out` 与它即将填充的窗口有重叠。`check_window_aliasing` 比较的是区间,因此也覆盖
了 `y = a2a_borrowed(x, tag="t"); a2a_borrowed(y, mode=1, tag="t")` 这种往返 —— 这是调用方很自然
就会写出来的形状,而且它曾经因为旧的内存池 key 而**碰巧**能工作。

## 异步结果

`all_to_all_single_4d_async` 返回一个注册进 torch work registry 的 `AsyncCollectiveTensor`,所以
对结果做的第一个 aten 算子会自己等待。registry 的 key 是输出的 **storage**,不是它的地址:每次调用
的 `at::from_blob` 视图都带一个新的 storage,所以一条 registry 记录属于**那次调用**,永远不属于它
所别名的那个窗口。

当链接进来的 libtorch 里没有这个 registry 时(构建期一次 `nm` 探测决定,
`build_info()["has_work_registry"]` 会报出来),同样这些函数返回一个带显式 `.wait()` 的 handle。
一个被直接丢弃的结果会在 registry 里留下记录,torch 会在进程退出时打印幸存记录的数量。

同步的集合操作留在调用者的流上,而不是通信流上。把它们绕到通信流要每次调用多付两次 event 跳转,
量级和这次集合通信本身相当。
