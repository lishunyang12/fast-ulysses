#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ulysses {

class RdmaBuffer {
public:
    ~RdmaBuffer();

private:
    friend class RdmaTransport;
    struct Impl;
    explicit RdmaBuffer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class RdmaTransport {
public:
    RdmaTransport(int rank, int world_size, int device,
                  const std::vector<int64_t>& devices);
    ~RdmaTransport();

    bool enabled() const;
    const std::string& nic() const;
    std::vector<int64_t> connection_info() const;
    void connect(const std::vector<std::vector<int64_t>>& peers);

    std::unique_ptr<RdmaBuffer> register_buffer(void* pointer,
                                                int64_t bytes,
                                                int mode,
                                                int64_t batch,
                                                int64_t seq,
                                                int64_t heads,
                                                int64_t dim,
                                                int64_t element_size);
    std::vector<int64_t> buffer_info(const RdmaBuffer& buffer) const;
    void connect_buffer(RdmaBuffer& buffer,
                        const std::vector<std::vector<int64_t>>& peers) const;
    std::vector<uint64_t> peer_pointers(const RdmaBuffer& buffer) const;
    void exchange(const void* input,
                  int64_t input_bytes,
                  RdmaBuffer& output,
                  int mode,
                  int64_t batch,
                  int64_t seq,
                  int64_t heads,
                  int64_t dim,
                  int64_t element_size);
    void flush() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ulysses
