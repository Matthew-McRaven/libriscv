#pragma once
#include <string>
#include <unordered_set>
template <AddressType address_t>
static std::vector<riscv::address_t> load_jump_hints(const std::string& filename, bool verbose = false);
template <AddressType address_t>
static void store_jump_hints(const std::string& filename, const std::vector<riscv::address_t>& hints);

#if defined(EMULATOR_MODE_LINUX)
	static constexpr bool full_linux_guest = true;
#else
	static constexpr bool full_linux_guest = false;
#endif
#if defined(EMULATOR_MODE_NEWLIB)
	static constexpr bool newlib_mini_guest = true;
#else
	static constexpr bool newlib_mini_guest = false;
#endif
#if defined(EMULATOR_MODE_MICRO)
	static constexpr bool micro_guest = true;
#else
	static constexpr bool micro_guest = false;
#endif
