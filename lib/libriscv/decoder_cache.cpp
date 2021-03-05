#include "memory.hpp"
#include "machine.hpp"
#include "decoder_cache.hpp"
#include <stdexcept>

#include "rv32i_instr.hpp"

namespace riscv
{
	static const size_t PAGE_SIZE = 4096;
	inline size_t page_number(uint64_t addr) {
		return addr >> 12;
	}

#ifdef RISCV_INSTR_CACHE
	template <int W>
	void Memory<W>::generate_decoder_cache(const MachineOptions<W>& options,
		address_t pbase, address_t addr, size_t len)
	{
		constexpr address_t PMASK = PAGE_SIZE-1;
		const size_t prelen  = addr - pbase;
		const size_t midlen  = len + prelen;
		const size_t plen =
			(PMASK & midlen) ? ((midlen + PAGE_SIZE) & ~PMASK) : midlen;

		const size_t n_pages = plen / PAGE_SIZE;
		auto* decoder_array = new DecoderCache<W> [n_pages];
		this->m_exec_decoder =
			decoder_array[0].get_base() - pbase / decoder_array->DIVISOR;
		// there could be an old cache from a machine reset
		delete[] this->m_decoder_cache;
		this->m_decoder_cache = &decoder_array[0];

#ifdef RISCV_INSTR_CACHE_PREGEN
		std::vector<typename CPU<W>::instr_pair> ipairs;

		auto* exec_offset = m_exec_pagedata.get() - pbase;
		// generate instruction handler pointers for machine code
		for (address_t dst = pbase; dst < pbase + plen;)
		{
			const size_t cacheno = page_number(dst - pbase);
			const address_t offset = dst & (PAGE_SIZE-1);
			auto& cache = decoder_array[cacheno];
			auto& entry = cache.get(offset / cache.DIVISOR);

			if (dst >= addr && dst < addr + len)
			{
				auto& instruction = *(rv32i_instruction*) &exec_offset[dst];
			#ifdef RISCV_DEBUG
				ipairs.emplace_back(entry.handler, instruction);
			#else
				ipairs.emplace_back(entry, instruction);
			#endif

				cache.convert(machine().cpu.decode(instruction), entry);
			} else {
				cache.convert(machine().cpu.decode({0}), entry);
			}
			if constexpr (compressed_enabled)
				dst += 2; /* We need to handle all entries */
			else
				dst += 4;
		}

#ifdef RISCV_BINARY_TRANSLATION
		machine().cpu.try_translate(options, addr, ipairs);
#endif
		for (size_t n = 0; n < ipairs.size()-1; n++)
		{
			if (machine().cpu.try_fuse(ipairs[n+0], ipairs[n+1]))
				n += 1;
		}

#else
		// zero the whole thing
		std::memset(decoder_array, 0, n_pages * sizeof(decoder_array[0]));
#endif
		(void) options;
	}
#endif

#ifndef __GNUG__
	template struct Memory<4>;
	template struct Memory<8>;
#endif
}
