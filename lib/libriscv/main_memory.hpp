#pragma once
#include <cstdint>
#include "types.hpp"

namespace riscv
{
	template<int W>
	struct MainMemory {
		using address_t = address_type<W>;

		address_t size() const noexcept { return physend - physbase; }

		bool within_ro(address_t addr, size_t asize) const noexcept {
			return (addr >= robase) && (addr + asize <= this->physend);
		}
		bool within_rw(address_t addr, size_t asize) const noexcept {
			return (addr >= rwbase) && (addr + asize <= this->physend);
		}
		const char* ro_at(address_t addr, size_t asize) const {
			if (LIKELY(within_ro(addr, asize)))
				return &mem[addr - physbase];
			throw MachineException(PROTECTION_FAULT, "MainMemory::ro_at() invalid address", addr);
		}
		char* rw_at(address_t addr, size_t asize) {
			if (LIKELY(within_rw(addr, asize)))
				return &mem[addr - physbase];
			throw MachineException(PROTECTION_FAULT, "MainMemory::rw_at() invalid address", addr);
		}

		bool unsafe_within(address_t addr, size_t asize) const noexcept {
			return (addr >= physbase) && (addr + asize <= this->physend);
		}
		char* unsafe_at(address_t addr, size_t asize) {
			if (LIKELY(unsafe_within(addr, asize)))
				return &mem[addr - physbase];
			throw MachineException(PROTECTION_FAULT, "MainMemory::unsafe_at() invalid address", addr);
		}
		const char* unsafe_at(address_t addr, size_t asize) const {
			if (LIKELY(unsafe_within(addr, asize)))
				return &mem[addr - physbase];
			throw MachineException(PROTECTION_FAULT, "MainMemory::unsafe_at() invalid address", addr);
		}

		size_t max_length(address_t addr) const noexcept {
			if (LIKELY(addr >= physbase && addr < physend))
				return physend - addr;
			return 0;
		}

		MainMemory(size_t size);
		MainMemory(const MainMemory& other);
		~MainMemory();

		char* mem = nullptr;
		const address_t physbase;
		const address_t physend;

		address_t robase = 0;
		address_t rwbase = 0;
		int fd = 0;
	};

} // riscv
