#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <fast_ulysses/common.hpp>
#include <fast_ulysses/group.hpp>
#include <fast_ulysses/transfer.hpp>

#include <set>

namespace ulysses {
namespace symm = c10d::symmetric_memory;

namespace {

int64_t flag_offset(int world_size)
{
    const int64_t slots = static_cast<int64_t>(symm::get_signal_pad_size()) / 8;
    TORCH_CHECK(slots >= world_size + 8,
                "symmetric-memory signal pad is too small for ", world_size,
                " flags");
    return slots - world_size;
}

bool all_pairs_support(const std::vector<int64_t>& devices, bool atomics)
{
    for (int64_t src : devices) {
        for (int64_t dst : devices) {
            if (src == dst) continue;
            int value = 0;
            if (atomics) {
                ULYSSES_CUDA_CHECK(cudaDeviceGetP2PAttribute(
                    &value, cudaDevP2PAttrNativeAtomicSupported,
                    static_cast<int>(src), static_cast<int>(dst)));
            } else {
                ULYSSES_CUDA_CHECK(cudaDeviceCanAccessPeer(
                    &value, static_cast<int>(src), static_cast<int>(dst)));
            }
            if (!value) return false;
        }
    }
    return true;
}

}  // namespace

UlyssesGroup::UlyssesGroup(std::string group_name,
                           int64_t rank,
                           int64_t world_size,
                           int64_t device_index,
                           std::vector<int64_t> devices)
    : group_name_(std::move(group_name)),
      rank_(static_cast<int>(rank)),
      world_size_(static_cast<int>(world_size)),
      device_index_(static_cast<int>(device_index))
{
    TORCH_CHECK(world_size_ >= 1 && world_size_ <= 8,
                "world_size must be in [1, 8]");
    TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "rank is out of range");
    TORCH_CHECK(static_cast<int>(devices.size()) == world_size_,
                "device list must contain one entry per rank");
    TORCH_CHECK(static_cast<int>(std::set<int64_t>(devices.begin(), devices.end()).size()) ==
                    world_size_,
                "one rank per GPU is required");
    TORCH_CHECK(devices[rank_] == device_index_,
                "rank device does not match the gathered device list");
    TORCH_CHECK(all_pairs_support(devices, false),
                "every GPU pair must support CUDA peer access");
    device_barrier_ = all_pairs_support(devices, true);
}

UlyssesGroup::~UlyssesGroup()
{
    destroy();
}

void UlyssesGroup::validate_input(const at::Tensor& input, int64_t mode) const
{
    TORCH_CHECK(!destroyed_, "UlyssesGroup has been destroyed");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.get_device() == device_index_, "input is on the wrong GPU");
    TORCH_CHECK(input.dim() == 4, "input must be [B, S, H, D]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kHalf ||
                    input.scalar_type() == at::kBFloat16,
                "input dtype must be float16 or bfloat16");
    TORCH_CHECK(!input.requires_grad(), "fast-ulysses minimal is inference-only");
    for (int64_t size : input.sizes()) {
        TORCH_CHECK(size > 0, "all input dimensions must be positive");
    }
    const int64_t split_axis = mode == 0 ? input.size(2) : input.size(1);
    TORCH_CHECK(split_axis % world_size_ == 0,
                "the scattered axis must divide world_size");
}

std::vector<int64_t> UlyssesGroup::output_shape(const at::Tensor& input,
                                                 int64_t mode) const
{
    validate_input(input, mode);
    const int64_t b = input.size(0);
    const int64_t s = input.size(1);
    const int64_t h = input.size(2);
    const int64_t d = input.size(3);
    if (mode == 0) return {b, s * world_size_, h / world_size_, d};
    return {b, s / world_size_, h * world_size_, d};
}

at::Tensor UlyssesGroup::allocate_output(const at::Tensor& input, int64_t mode)
{
    const std::vector<int64_t> shape = output_shape(input, mode);
    const int64_t numel = input.numel();
    const at::cuda::CUDAGuard guard(device_index_);
    at::Tensor tensor = symm::empty_strided_p2p(
        {numel}, {1}, input.scalar_type(),
        c10::Device(c10::DeviceType::CUDA, device_index_), group_name_, std::nullopt);
    auto memory = symm::rendezvous(tensor, group_name_);
    TORCH_CHECK(memory, "symmetric-memory rendezvous failed for group '",
                group_name_, "'");

    auto buffer = std::make_unique<Buffer>();
    buffer->tensor = tensor;
    buffer->shape = shape;
    buffer->dtype = input.scalar_type();
    for (void* ptr : memory->get_buffer_ptrs())
        buffer->peer_ptrs.push_back(reinterpret_cast<uint64_t>(ptr));
    const int64_t offset = flag_offset(world_size_);
    for (void* ptr : memory->get_signal_pad_ptrs())
        buffer->flag_ptrs.push_back(
            reinterpret_cast<uint64_t>(ptr) + static_cast<uint64_t>(offset * 8));
    TORCH_CHECK(static_cast<int>(buffer->peer_ptrs.size()) == world_size_,
                "symmetric-memory group size mismatch");

    memory->get_signal_pad(rank_, {world_size_}, at::kLong, offset).zero_();

    at::Tensor out = tensor.view(shape);
    Buffer* raw = buffer.get();
    buffers_.push_back(std::move(buffer));
    by_address_[out.data_ptr()] = raw;
    return out;
}

Buffer& UlyssesGroup::find_buffer(const at::Tensor& out)
{
    const auto it = by_address_.find(out.data_ptr());
    TORCH_CHECK(it != by_address_.end(),
                "out must come from this group's allocate_output()");
    return *it->second;
}

void UlyssesGroup::exchange(const at::Tensor& input,
                            at::Tensor out,
                            int64_t mode,
                            int64_t stream_ptr)
{
    const std::vector<int64_t> expected = output_shape(input, mode);
    Buffer& buffer = find_buffer(out);
    TORCH_CHECK(out.is_cuda() && out.get_device() == device_index_,
                "out is on the wrong GPU");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out.scalar_type() == input.scalar_type(), "dtype mismatch");
    TORCH_CHECK(out.sizes().vec() == expected, "out has the wrong shape");
    TORCH_CHECK(buffer.shape == expected && buffer.dtype == input.scalar_type(),
                "out was allocated for a different exchange");
    TORCH_CHECK(input.data_ptr() != out.data_ptr(), "input and out must not alias");

    const at::cuda::CUDAGuard guard(device_index_);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (device_barrier_) fast_barrier(stream, buffer.flag_ptrs, rank_, ++buffer.epoch);
    launch_equal_a2a(input.data_ptr(), buffer.peer_ptrs, static_cast<int>(mode),
                     input.size(0), input.size(1), input.size(2), input.size(3),
                     static_cast<int64_t>(input.element_size()), rank_, stream);
    if (device_barrier_) fast_barrier(stream, buffer.flag_ptrs, rank_, ++buffer.epoch);
}

void UlyssesGroup::destroy()
{
    if (destroyed_) return;
    destroyed_ = true;
    by_address_.clear();
    buffers_.clear();
}

}  // namespace ulysses
