#pragma once
#include <cstdint>
#include <memory>
#include "types.hpp"

namespace riscv
{
	template<int W>
	struct MainMemory {
		MainMemory(size_t size);

		address_type<W> size() const noexcept { return physend - physbase; }

		bool within_ro(uint64_t addr, size_t asize) const noexcept {
			return (addr >= robase) && (addr + asize <= this->physend);
		}
		bool within_rw(uint64_t addr, size_t asize) const noexcept {
			return (addr >= rwbase) && (addr + asize <= this->physend);
		}
		const char* ro_at(uint64_t addr, size_t asize) const {
			if (within_ro(addr, asize))
				return &mem[addr - physbase];
			throw MachineException(PROTECTION_FAULT, "MainMemory::ro_at() invalid address", addr);
		}
		char* rw_at(uint64_t addr, size_t asize) {
			if (within_rw(addr, asize))
				return &mem[addr - physbase];
			throw MachineException(PROTECTION_FAULT, "MainMemory::rw_at() invalid address", addr);
		}

		bool unsafe_within(uint64_t addr, size_t asize) const noexcept {
			return (addr >= physbase) && (addr + asize <= this->physend);
		}
		char* unsafe_at(uint64_t addr, size_t asize) {
			if (unsafe_within(addr, asize))
				return &mem[addr - physbase];
			throw MachineException(PROTECTION_FAULT, "MainMemory::unsafe_at() invalid address", addr);
		}
		const char* unsafe_at(uint64_t addr, size_t asize) const {
			if (unsafe_within(addr, asize))
				return &mem[addr - physbase];
			throw MachineException(PROTECTION_FAULT, "MainMemory::unsafe_at() invalid address", addr);
		}

		size_t max_length(uint64_t addr) const noexcept {
			if (addr >= physbase && addr < physend)
				return physend - addr;
			return 0;
		}

		std::unique_ptr<char[]> mem = nullptr;
		const address_type<W> physbase;
		const address_type<W> physend;

		address_type<W> robase = 0;
		address_type<W> rwbase = 0;
	};

	template <int W> inline
	MainMemory<W>::MainMemory(size_t size)
		: physbase(0), physend(size)
	{
		mem.reset(new char[size]);
	}

} // riscv
