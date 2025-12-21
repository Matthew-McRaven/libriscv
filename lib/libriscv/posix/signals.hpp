#pragma once
#include <map>
#include <set>
#include "../core/registers.hpp"
#include "../riscvbase.hpp"
#include "../types.hpp"

namespace riscv {
template <AddressType> struct Machine;
template <AddressType> struct Registers;

template <AddressType address_t> struct SignalStack {
  address_t ss_sp = 0x0;
  int ss_flags = 0x0;
  address_t ss_size = 0;
};

template <AddressType address_t> struct SignalAction {
  static constexpr address_t SIG_UNSET = ~(address_t)0x0;
  bool is_unset() const noexcept { return handler == 0x0 || handler == SIG_UNSET; }
  address_t handler = SIG_UNSET;
  bool altstack = false;
  unsigned mask = 0x0;
};

template <AddressType address_t> struct SignalReturn {
  Registers<address_t> regs;
};

template <AddressType address_t> struct SignalPerThread {
  SignalStack<address_t> stack;
  SignalReturn<address_t> sigret;
};

template <AddressType address_t> struct Signals {
  SignalAction<address_t> &get(int sig);
  void enter(Machine<address_t> &, int sig);

  // TODO: Lock this in the future, for multiproessing
  auto& per_thread(int tid) { return m_per_thread[tid]; }

	Signals();
	~Signals();
private:
  std::array<SignalAction<address_t>, 64> signals{};
  std::map<int, SignalPerThread<address_t>> m_per_thread;
};

template <AddressType address_t> Signals<address_t>::Signals() {}
template <AddressType address_t> Signals<address_t>::~Signals() {}

template <AddressType address_t> SignalAction<address_t> &Signals<address_t>::get(int sig) {
  if (sig > 0) return signals.at(sig - 1);
  throw MachineException(ILLEGAL_OPERATION, "Signal 0 invoked");
}

} // riscv
