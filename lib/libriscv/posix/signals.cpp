#include "signals.hpp"
#include "../common.hpp"
#include "../machine.hpp"

namespace riscv {
template <AddressType address_t> Signals<address_t>::Signals() {}
template <AddressType address_t> Signals<address_t>::~Signals() {}

template <AddressType address_t> SignalAction<address_t> &Signals<address_t>::get(int sig) {
  if (sig > 0) return signals.at(sig - 1);
  throw MachineException(ILLEGAL_OPERATION, "Signal 0 invoked");
}

template <AddressType address_t> void Signals<address_t>::enter(Machine<address_t> &machine, int sig) {
  if (sig == 0) return;

  auto &sigact = signals.at(sig);
  if (sigact.altstack) {
    auto *thread = machine.threads().get_thread();
    // Change to alternate per-thread stack
    auto &stack = per_thread(thread->tid).stack;
    machine.cpu.reg(REG_SP) = stack.ss_sp + stack.ss_size;
  }
  // We have to jump to handler-4 because we are mid-instruction
  // WARNING: Assumption.
  machine.cpu.jump(sigact.handler - 4);
}
template struct Signals<uint32_t>;
template struct Signals<uint64_t>;
} // namespace riscv
