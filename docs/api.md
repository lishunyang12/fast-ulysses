# API

## `UlyssesGroup(process_group=None, device=None)`

Construction gathers the CUDA device for every rank. It requires one rank per GPU and CUDA P2P
access for every ordered pair. Native peer atomics select the device-barrier backend; otherwise the
group uses the blocking PCIe backend.

## `allocate_output(x, mode=0)`

Collectively allocates a symmetric output tensor. Every rank must call allocations in the same
order. Keep and reuse the result; buffers remain owned by the group until `destroy()`.

## `exchange(x, out, mode=0, stream=None)`

Writes directly into an output returned by the same group's `allocate_output`. Inputs and outputs
must be contiguous CUDA FP16/BF16 tensors. Gradients, uneven splits and aliasing are rejected.

On the device backend, work is queued on `stream` (or the current stream) and the call returns after
submission. On PCIe, the stream and process group are synchronized before and after the copy.

## `destroy()`

Collectively synchronizes and releases all symmetric buffers.
