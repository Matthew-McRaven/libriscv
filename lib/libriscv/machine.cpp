#include "./machine.hpp"
#include "./machine_impl.hpp"
#include "./machine_threads.hpp"
#include "./machine_vmcall.hpp"
#include "./native_heap.hpp"

template struct riscv::Machine<uint32_t>;
template struct riscv::Machine<uint64_t>;
