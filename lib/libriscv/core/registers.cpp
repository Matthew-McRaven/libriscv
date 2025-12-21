#include "./registers.hpp"
#include "../riscvbase.hpp"

namespace riscv {

template <AddressType address_t> RISCV_COLD_PATH() std::string Registers<address_t>::to_string() const {
  char buffer[600];
  int len = 0;
  for (int i = 1; i < 32; i++) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%08X] ", RISCV::regname(i), this->get(i));
    if (i % 5 == 4) len += snprintf(buffer + len, sizeof(buffer) - len, "\n");
  }
  return std::string(buffer, len);
}

template <AddressType address_t> RISCV_COLD_PATH() std::string Registers<address_t>::flp_to_string() const {
  char buffer[800];
  int len = 0;
  for (int i = 0; i < 32; i++) {
    auto &src = this->getfl(i);
    const char T = (src.i32[1] == 0) ? 'S' : 'D';
    if constexpr (true) {
      double val = (src.i32[1] == 0) ? src.f32[0] : src.f64;
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%c%+.2f] ", RISCV::flpname(i), T, val);
    } else {
      if (src.i32[1] == 0) {
        double val = src.f64;
        len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%c0x%lX] ", RISCV::flpname(i), T, *(int64_t *)&val);
      } else {
        float val = src.f32[0];
        len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%c0x%X] ", RISCV::flpname(i), T, *(int32_t *)&val);
      }
    }
    if (i % 5 == 4) {
      len += snprintf(buffer + len, sizeof(buffer) - len, "\n");
    }
  }
  len += snprintf(buffer + len, sizeof(buffer) - len, "[FFLAGS\t0x%X] ", m_fcsr.fflags);
  return std::string(buffer, len);
}

} // namespace riscv

template struct riscv::Registers<uint32_t>;
template struct riscv::Registers<uint64_t>;
