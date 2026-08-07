#pragma once
// Binds a result that completes on the comm stream to torch's functional-collective
// machinery, so an async call can hand back a
// torch.distributed._functional_collectives.AsyncCollectiveTensor: a wrapper whose .wait()
// -- or whose first use by ANY aten op -- inserts the cross-stream dependency. Forgetting
// to wait stops being a silent read of a buffer the comm stream is still writing.
//
// torch.ops._c10d_functional.wait_tensor looks a tensor up in a registry and calls wait()
// on every c10d::Work it finds there. Registering ours is what gives callers those standard
// semantics instead of a bespoke handle they have to remember to await.
//
// The registry keys by the output's STORAGE, not by its address: measured on torch 2.13, a
// second tensor built over the same memory with its own storage does not find the work, and
// two works on ONE storage are both waited by a single wait_tensor. What that means for the
// borrowed entry point, whose result is an at::from_blob view of the symmetric window: each
// call's view carries a fresh storage, so an entry belongs to that CALL, never to the window
// -- waiting on call N's result waits on call N's event and says nothing about whether call
// N+1 has already overwritten the bytes. That is the borrowed lifetime contract unchanged,
// not a new hazard.
#include <ATen/ATen.h>
#include <cuda_runtime.h>

namespace ulysses {

// Records a completion event on `comm_stream` and binds it to `tensor`. Call it with the
// CALLER's stream current -- the fallback below inserts its wait there.
//
// Returns true when the event is now the registry's, to wait on and to destroy.
//
// Returns false when this build has no registry (FAST_ULYSSES_HAS_WORK_REGISTRY=0): nothing
// could ever wait on the event through torch, so the caller's current stream is made to wait
// on it before returning and the event is destroyed. The result is then already safe to read
// -- correct, only without the overlap the async path exists for -- and comm.py returns a
// CompletedHandle rather than a wrapper nobody would wait on.
bool register_stream_completion(const at::Tensor& tensor, cudaStream_t comm_stream);

}  // namespace ulysses
