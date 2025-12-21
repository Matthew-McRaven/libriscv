#pragma once
#include "../machine.hpp"

namespace riscv {
// machine_defaults.cpp
// Default: Stdout allowed
template <AddressType address_t>
void Machine<address_t>::default_printer(const Machine<address_t> &, const char *buffer, size_t len) {
  std::ignore = ::write(1, buffer, len);
}
// Default: Stdin *NOT* allowed
template <AddressType address_t>
long Machine<address_t>::default_stdin(const Machine<address_t> &, char * /*buffer*/, size_t /*len*/) {
  return 0;
}

// Default: RDTIME produces monotonic time with *microsecond*-granularity
template <AddressType address_t> uint64_t Machine<address_t>::default_rdtime(const Machine<address_t> &machine) {
#ifdef __wasm__
  return 0;
#else
  auto now = std::chrono::steady_clock::now();
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  if (!(machine.has_file_descriptors() && machine.fds().proxy_mode)) micros &= ANTI_FINGERPRINTING_MASK_MICROS();
  return micros;
#endif
}

} // namespace riscv
