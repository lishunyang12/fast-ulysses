#include <torch/extension.h>

#include <fast_ulysses/group.hpp>

#ifndef FAST_ULYSSES_VERSION
#define FAST_ULYSSES_VERSION "unknown"
#endif
#ifndef FAST_ULYSSES_CUDA_ARCH_LIST
#define FAST_ULYSSES_CUDA_ARCH_LIST "unknown"
#endif

TORCH_LIBRARY(fast_ulysses, m)
{
    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::string, int64_t, int64_t, int64_t,
                         std::vector<int64_t>>())
        .def("allocate_output", &ulysses::UlyssesGroup::allocate_output)
        .def("exchange", &ulysses::UlyssesGroup::exchange)
        .def("uses_device_barrier", &ulysses::UlyssesGroup::uses_device_barrier)
        .def("backend", &ulysses::UlyssesGroup::backend)
        .def("destroy", &ulysses::UlyssesGroup::destroy);
}

PYBIND11_MODULE(_C, m)
{
    m.def("build_info", []() {
        pybind11::dict out;
        out["version"] = FAST_ULYSSES_VERSION;
        out["cuda_arch_list"] = FAST_ULYSSES_CUDA_ARCH_LIST;
        return out;
    });
}
