#include "signals.hpp"
#include "../common.hpp"

namespace riscv {
	INSTANTIATE_32_IF_ENABLED(Signals);
	INSTANTIATE_64_IF_ENABLED(Signals);
} // riscv
