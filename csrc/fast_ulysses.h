#pragma once

#include "rdma.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <torch/custom_class.h>
#include <torch/version.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

static_assert(TORCH_VERSION_MAJOR > 2 ||
                  (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10),
              "fast-ulysses requires torch 2.10 or newer");

#define FU_CUDA_CHECK(expr)                                                    \
    do {                                                                       \
        const cudaError_t err = (expr);                                        \
        TORCH_CHECK(err == cudaSuccess, #expr, ": ", cudaGetErrorString(err)); \
    } while (0)

namespace ulysses {

struct Buffer {
    at::Tensor tensor;
    at::Tensor stage;
    std::vector<uint64_t> peers;
    std::vector<uint64_t> stage_peers;
    std::vector<uint64_t> flags;
    std::vector<int64_t> shape;
    at::ScalarType dtype = at::ScalarType::Undefined;
    uint64_t epoch = 0;
    std::unique_ptr<RdmaBuffer> rdma;
};

class UlyssesGroup final : public torch::CustomClassHolder {
public:
    UlyssesGroup(std::string name,
                 int64_t rank,
                 int64_t world_size,
                 int64_t device,
                 std::vector<int64_t> devices);
    ~UlyssesGroup() override;

    at::Tensor allocate_output(const at::Tensor& input, int64_t mode);
    void release_output(at::Tensor output);
    void exchange(const at::Tensor& input,
                  at::Tensor output,
                  int64_t mode,
                  int64_t stream);
    std::string backend() const;
    std::vector<int64_t> connection_info() const;
    void connect(const std::vector<std::vector<int64_t>>& peers);
    std::vector<int64_t> buffer_info(at::Tensor output) const;
    void connect_buffer(at::Tensor output,
                        const std::vector<std::vector<int64_t>>& peers);
    void flush() const;
    void destroy();

private:
    void validate(const at::Tensor& input, int64_t mode) const;
    std::vector<int64_t> output_shape(const at::Tensor& input, int64_t mode) const;

    std::string name_;
    int rank_;
    int world_size_;
    int device_;
    bool destroyed_ = false;
    std::unique_ptr<RdmaTransport> rdma_;
    std::vector<std::unique_ptr<Buffer>> buffers_;
    std::map<const void*, Buffer*> outputs_;
    std::optional<c10::cuda::CUDAStream> pack_stream_;
    std::optional<c10::cuda::CUDAStream> send_stream_;
    std::vector<cudaEvent_t> events_;
};

// The dimensions of one exchange. `mode` 0 splits heads and gathers sequence, `mode` 1 the
// reverse; `seq` and `heads` are always the input's, so one of them is the local half.
struct Dims {
    int mode;
    int rank;
    int world_size;
    int64_t batch;
    int64_t seq;
    int64_t heads;
    int64_t dim;
    int64_t element_size;
};

// Bytes this rank exchanges with one peer.
int64_t slice_bytes(const Dims& dims);

// Moves this rank's share to every peer named in `peers` and its own share to `output`.
// A peer copy carries flat runs only: the relayout is a device-local copy, which costs an
// order of magnitude less per byte than a strided copy across a link.
void exchange_peers(const void* input,
                    void* output,
                    void* stage,
                    const std::vector<uint64_t>& output_peers,
                    const std::vector<uint64_t>& stage_peers,
                    const std::vector<int>& peers,
                    const Dims& dims,
                    cudaStream_t caller,
                    cudaStream_t pack_stream,
                    cudaStream_t send_stream,
                    const std::vector<cudaEvent_t>& events);

// Strided copies straight into the peers' outputs, with no staging buffer. Kept for the mlx5
// backend, whose outputs are plain allocations with no symmetric staging window.
void exchange_peers_direct(const void* input,
                           const std::vector<uint64_t>& output_peers,
                           const std::vector<int>& peers,
                           const Dims& dims,
                           cudaStream_t stream);

// `mode` 1 only: scatter what the peers left in `stage` into the head blocks they own.
// Ordered after the closing barrier, because it reads what the peers wrote.
void unpack_peers(void* output,
                  const void* stage,
                  const std::vector<int>& peers,
                  const Dims& dims,
                  cudaStream_t stream);

void barrier(cudaStream_t stream,
             const std::vector<uint64_t>& flags,
             int rank,
             uint64_t epoch);

}  // namespace ulysses
