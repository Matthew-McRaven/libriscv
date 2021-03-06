#pragma once
#include <cstdint>

namespace riscv {
	template <int W>
	struct CallbackTable {
		uint32_t (*mem_read32)(CPU<W>&, address_type<W> addr);
		uint64_t (*mem_read64)(CPU<W>&, address_type<W> addr);
		void (*jump)(CPU<W>&, address_type<W>, uint64_t);
		int  (*syscall)(CPU<W>&, address_type<W>, uint64_t);
		void (*stop)(CPU<W>&, uint64_t);
		void (*ebreak)(CPU<W>&, uint64_t);
		void (*system)(CPU<W>&, uint32_t);
		void (*trigger_exception)(CPU<W>&, int, address_type<W>);
		float  (*sqrtf32)(float);
		double (*sqrtf64)(double);
	};
}
