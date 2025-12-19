#include "signals.hpp"
#include "../internal_common.hpp"

namespace riscv {
	INSTANTIATE_32_IF_ENABLED(Signals);
	INSTANTIATE_64_IF_ENABLED(Signals);
	INSTANTIATE_128_IF_ENABLED(Signals);
} // riscv
