#include "../common.hpp"
#include "./cpu.hpp"

namespace riscv {
template <AddressType address_t>
inline CPU<address_t>::CPU(Machine<address_t> &machine) : m_machine{machine}, m_exec(empty_execute_segment().get()) {}
template <AddressType address_t> inline void CPU<address_t>::reset_stack_pointer() noexcept {
  // initial stack location
  this->reg(2) = machine().memory.stack_initial();
}

template <AddressType address_t> inline void CPU<address_t>::jump(const address_t dst) {
  // it's possible to jump to a misaligned address
  if constexpr (!compressed_enabled) {
    if (UNLIKELY(dst & 0x3)) {
      trigger_exception(MISALIGNED_INSTRUCTION, dst);
    }
  } else {
    if (UNLIKELY(dst & 0x1)) {
      trigger_exception(MISALIGNED_INSTRUCTION, dst);
    }
  }
  this->registers().pc = dst;
}

template <AddressType address_t> inline void CPU<address_t>::aligned_jump(const address_t dst) noexcept {
  this->registers().pc = dst;
}

template <AddressType address_t> inline void CPU<address_t>::increment_pc(int delta) noexcept {
  registers().pc += delta;
}

// Use a trick to access the Machine directly on g++/clang, Linux-only for now
#if (defined(__GNUG__) || defined(__clang__)) && defined(__linux__)
template <AddressType address_t> RISCV_ALWAYS_INLINE inline
Machine<address_t>& CPU<address_t>::machine() noexcept { return *reinterpret_cast<Machine<address_t>*> (this); }
template <AddressType address_t> RISCV_ALWAYS_INLINE inline
const Machine<address_t>& CPU<address_t>::machine() const noexcept { return *reinterpret_cast<const Machine<address_t>*> (this); }
#else
template <AddressType address_t> RISCV_ALWAYS_INLINE inline
Machine<address_t>& CPU<address_t>::machine() noexcept { return this->m_machine; }
template <AddressType address_t> RISCV_ALWAYS_INLINE inline
const Machine<address_t>& CPU<address_t>::machine() const noexcept { return this->m_machine; }
#endif

template <AddressType address_t> RISCV_ALWAYS_INLINE inline
Memory<address_t>& CPU<address_t>::memory() noexcept { return machine().memory; }
template <AddressType address_t> RISCV_ALWAYS_INLINE inline
const Memory<address_t>& CPU<address_t>::memory() const noexcept { return machine().memory; }
} // namespace riscv
