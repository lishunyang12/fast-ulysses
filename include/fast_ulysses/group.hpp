#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <torch/custom_class.h>
#include <vector>

namespace ulysses {

struct Buffer {
    at::Tensor tensor;
    std::vector<uint64_t> peer_ptrs;
    std::vector<uint64_t> flag_ptrs;
    std::vector<int64_t> shape;
    at::ScalarType dtype = at::kBFloat16;
    uint64_t epoch = 0;
};

class UlyssesGroup final : public torch::CustomClassHolder {
public:
    UlyssesGroup(std::string group_name,
                 int64_t rank,
                 int64_t world_size,
                 int64_t device_index,
                 std::vector<int64_t> devices);
    ~UlyssesGroup() override;

    at::Tensor allocate_output(const at::Tensor& input, int64_t mode);
    void exchange(const at::Tensor& input,
                  at::Tensor out,
                  int64_t mode,
                  int64_t stream_ptr);
    bool uses_device_barrier() const { return device_barrier_; }
    std::string backend() const { return device_barrier_ ? "device" : "pcie"; }
    void destroy();

private:
    std::vector<int64_t> output_shape(const at::Tensor& input, int64_t mode) const;
    void validate_input(const at::Tensor& input, int64_t mode) const;
    Buffer& find_buffer(const at::Tensor& out);

    std::string group_name_;
    int rank_ = 0;
    int world_size_ = 1;
    int device_index_ = 0;
    bool device_barrier_ = false;
    bool destroyed_ = false;
    std::vector<std::unique_ptr<Buffer>> buffers_;
    std::map<const void*, Buffer*> by_address_;
};

}  // namespace ulysses
