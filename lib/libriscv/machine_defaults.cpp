
#include "machine.hpp"
#include "internal_common.hpp"

#include <chrono>

namespace riscv
{

#ifndef __GNUG__ /* Workaround for GCC bug */
	INSTANTIATE_32_IF_ENABLED(Machine);
	INSTANTIATE_64_IF_ENABLED(Machine);
	INSTANTIATE_128_IF_ENABLED(Machine);
#endif

#ifdef RISCV_32I
  template void Machine<4>::setup_native_threads(const size_t);
#endif
#ifdef RISCV_64I
  template void Machine<8>::setup_native_threads(const size_t);
#endif
}
