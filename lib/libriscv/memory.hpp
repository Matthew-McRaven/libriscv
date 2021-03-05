#pragma once
#include "common.hpp"
#include "elf.hpp"
#include "main_memory.hpp"
#include <cassert>
#include <cstring>
#include "util/function.hpp"

namespace riscv
{
	template<int W> struct Machine;
	template<int W> struct DecoderCache;

	template<int W>
	struct Memory
	{
		using address_t = address_type<W>;

		template <typename T>
		T read(address_t src);

		template <typename T>
		T& writable_read(address_t src);

		template <typename T>
		void write(address_t dst, T value);

		void memset(address_t dst, uint8_t value, size_t len);
		void memcpy(address_t dst, const void* src, size_t);
		void memcpy(address_t dst, Machine<W>& srcm, address_t src, address_t len);
		void memcpy_out(void* dst, address_t src, size_t) const;
		// Gives a sequential view of the data at address, with the possibility
		// of optimizing away a copy if the data crosses no page-boundaries.
		std::string_view memview(address_t addr, size_t len) const;
		// Compare bounded memory
		int memcmp(address_t a1, address_t a2, size_t len) const;
		int memcmp(const void* h1, address_t a2, size_t len) const;
		// Read a zero-terminated string directly from guests memory
		std::string memstring(address_t addr, size_t maxlen = 1024) const;
		size_t strlen(address_t addr, size_t maxlen = 4096) const;

		address_t start_address() const noexcept { return this->m_start_address; }
		address_t stack_initial() const noexcept { return this->m_stack_address; }
		void set_stack_initial(address_t addr) { this->m_stack_address = addr; }
		address_t program_end() const noexcept { return this->m_program_end; }
		address_t max_memory() const noexcept { return m_main_memory.size(); }

		auto& machine() { return this->m_machine; }
		const auto& machine() const { return this->m_machine; }

		// Call interface
		address_t resolve_address(const std::string& sym) const;
		address_t resolve_section(const char* name) const;
		address_t exit_address() const noexcept;
		void      set_exit_address(address_t new_exit);
		// Basic backtraces and symbol lookups
		struct Callsite {
			std::string name = "(null)";
			address_t   address = 0x0;
			uint32_t    offset  = 0x0;
			size_t      size    = 0;
		};
		Callsite lookup(address_t) const;
		void print_backtrace(void(*print_function)(const char*, size_t));

		// Main memory (faster than pages)
		void create_main_memory(address_t base, size_t size);
		auto& main_memory() { return m_main_memory; }
		const auto& main_memory() const { return m_main_memory; }

#ifdef RISCV_INSTR_CACHE
		void generate_decoder_cache(const MachineOptions<W>&, address_t pbase, address_t va, size_t len);
		auto* get_decoder_cache() const { return m_exec_decoder; }
#endif

		const auto& binary() const noexcept { return m_binary; }
		void reset();

		bool is_binary_translated() const { return m_bintr_dl != nullptr; }
		void set_binary_translated(void* dl) const { m_bintr_dl = dl; }

		// serializes all the machine state + a tiny header to @vec
		void serialize_to(std::vector<uint8_t>& vec);
		// returns the machine to a previously stored state
		void deserialize_from(const std::vector<uint8_t>&, const SerializedMachine<W>&);

		Memory(Machine<W>&, std::string_view, MachineOptions<W>);
		~Memory();
	private:
		// ELF stuff
		using Ehdr = typename Elf<W>::Ehdr;
		using Phdr = typename Elf<W>::Phdr;
		using Shdr = typename Elf<W>::Shdr;
		template <typename T> T* elf_offset(intptr_t ofs) const {
			return (T*) &m_binary.at(ofs);
		}
		inline const auto* elf_header() const noexcept {
			return elf_offset<const Ehdr> (0);
		}
		const Shdr* section_by_name(const char* name) const;
		void relocate_section(const char* section_name, const char* symtab);
		const typename Elf<W>::Sym* resolve_symbol(const char* name) const;
		const auto* elf_sym_index(const Shdr* shdr, uint32_t symidx) const {
			assert(symidx < shdr->sh_size / sizeof(typename Elf<W>::Sym));
			auto* symtab = elf_offset<typename Elf<W>::Sym>(shdr->sh_offset);
			return &symtab[symidx];
		}
		// ELF loader
		void binary_loader(const MachineOptions<W>& options);
		void binary_load_ph(const MachineOptions<W>&, const Phdr*);
		// machine cloning
		void machine_loader(const Machine<W>&);
		void protection_fault(address_t);

		Machine<W>& m_machine;
		MainMemory<W> m_main_memory;
		const std::string_view m_binary;

		address_t m_start_address = 0;
		address_t m_stack_address = 0;
		address_t m_exit_address  = 0;
		address_t m_program_end = 0;
		const bool m_original_machine;

		// ELF programs linear .text segment
		std::unique_ptr<uint8_t[]> m_exec_pagedata = nullptr;
		size_t    m_exec_pagedata_size = 0;
		address_t m_exec_pagedata_base = 0;
#ifdef RISCV_INSTR_CACHE
	#ifdef RISCV_DEBUG
		using dchandler_t = Instruction<W>;
	#else
		using dchandler_t = instruction_handler<W>;
	#endif
		dchandler_t* m_exec_decoder = nullptr;
		DecoderCache<W>* m_decoder_cache = nullptr;
#endif
		mutable void* m_bintr_dl = nullptr;
	};
#include "memory_inline.hpp"
#include "memory_helpers.hpp"
}
