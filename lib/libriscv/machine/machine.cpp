#include "../machine.hpp"
#include "../native_heap.hpp"
#include "./machine_defaults.hpp"
#include "./machine_inline.hpp"
#include "./machine_serialize.hpp"
#include "./machine_threads.hpp"
#include "./machine_vmcall.hpp"

template struct riscv::Machine<uint32_t>;
template struct riscv::Machine<uint64_t>;
