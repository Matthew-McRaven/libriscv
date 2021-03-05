#include "memory.hpp"
#include <dlfcn.h>
#include <stdexcept>
#ifdef __GNUG__
#include "decoder_cache.cpp"
#endif
#include <sys/mman.h>

extern "C" char *
__cxa_demangle(const char *name, char *buf, size_t *n, int *status);

namespace riscv
{
	template <int W>
	Memory<W>::Memory(Machine<W>& mach, std::string_view bin,
					MachineOptions<W> options)
		: m_machine{mach},
		  m_main_memory{options.memory_max},
		  m_binary{bin},
		  m_original_machine {options.owning_machine == nullptr}
	{
		// when an owning machine is passed, its state will be used instead
		if (options.owning_machine == nullptr) {
			// load ELF binary into virtual memory
			if (!m_binary.empty())
				this->binary_loader(options);
		}
		else {
			this->machine_loader(*options.owning_machine);
		}
	}
	template <int W>
	Memory<W>::~Memory()
	{
#ifdef RISCV_INSTR_CACHE
		delete[] m_decoder_cache;
#endif
#ifdef RISCV_BINARY_TRANSLATION
		if (m_bintr_dl)
			dlclose(m_bintr_dl);
#endif
	}

	template <int W>
	void Memory<W>::reset()
	{
		// Hard to support because of things like
		// serialization, machine options and machine forks
	}

	template <int W>
	void Memory<W>::binary_load_ph(const MachineOptions<W>& options, const Phdr* hdr)
	{
		const auto*  src = m_binary.data() + hdr->p_offset;
		const size_t len = hdr->p_filesz;
		if (m_binary.size() <= hdr->p_offset ||
			hdr->p_offset + len <= hdr->p_offset)
		{
			throw std::runtime_error("Bogus ELF program segment offset");
		}
		if (m_binary.size() < hdr->p_offset + len) {
			throw std::runtime_error("Not enough room for ELF program segment");
		}
		if (hdr->p_vaddr + len < hdr->p_vaddr) {
			throw std::runtime_error("Bogus ELF segment virtual base");
		}

		if (options.verbose_loader) {
		printf("* Loading program of size %zu from %p to virtual %p\n",
				len, src, (void*) (uintptr_t) hdr->p_vaddr);
		}
		// segment permissions
		const struct {
			bool read, write, exec;
		} attr {
			.read  = (hdr->p_flags & PF_R) != 0,
			.write = (hdr->p_flags & PF_W) != 0,
			.exec  = (hdr->p_flags & PF_X) != 0
		};
		if (options.verbose_loader) {
		printf("* Program segment readable: %d writable: %d  executable: %d\n",
				attr.read, attr.write, attr.exec);
		}

		if (attr.exec && machine().cpu.exec_seg_data() == nullptr)
		{
			constexpr address_t PMASK = 4096-1;
			const address_t pbase = (hdr->p_vaddr - 0x4) & ~(address_t) PMASK;
			const size_t prelen  = hdr->p_vaddr - pbase;
			// The last 4 bytes of the range is zeroed out (illegal instruction)
			const size_t midlen  = len + prelen + 0x4;
			const size_t plen = midlen;
			const size_t postlen = plen - midlen;
			//printf("Addr 0x%X Len %zx becomes 0x%X->0x%X PRE %zx MIDDLE %zu POST %zu TOTAL %zu\n",
			//	hdr->p_vaddr, len, pbase, pbase + plen, prelen, len, postlen, plen);
			if (UNLIKELY(prelen > plen || prelen + len > plen)) {
				throw std::runtime_error("Segment virtual base was bogus");
			}
			// Create the whole executable memory range
			m_exec_pagedata.reset(new uint8_t[plen]);
			m_exec_pagedata_size = plen;
			m_exec_pagedata_base = pbase;
			std::memset(&m_exec_pagedata[0],      0,   prelen);
			std::memcpy(&m_exec_pagedata[prelen], src, len);
			std::memset(&m_exec_pagedata[prelen + len], 0,   postlen);
			// This is what the CPU instruction fetcher will use
			auto* exec_offset = m_exec_pagedata.get() - pbase;
			machine().cpu.initialize_exec_segs(exec_offset, hdr->p_vaddr, hdr->p_vaddr + len);
#if defined(RISCV_INSTR_CACHE)
			this->generate_decoder_cache(options, pbase, hdr->p_vaddr, len);
#endif
			(void) options;
			// Nothing more to do here, if execute-only
			if (!attr.read)
				return;
		} else if (attr.exec) {
#ifdef RISCV_INSTR_CACHE
			throw std::runtime_error(
				"Binary can not have more than one executable segment!");
#endif
		}
		// We would normally never allow this
		if (attr.exec && attr.write) {
			throw std::runtime_error("Insecure ELF has writable machine code");
		}

		if (attr.write) {
			if (m_main_memory.rwbase == 0 || m_main_memory.rwbase > hdr->p_vaddr)
				m_main_memory.rwbase = hdr->p_vaddr;
		}
		if (attr.read) {
			if (m_main_memory.robase == 0 || m_main_memory.robase > hdr->p_vaddr)
				m_main_memory.robase = hdr->p_vaddr;
		}

		std::memcpy(main_memory().unsafe_at(hdr->p_vaddr, len), src, len);
	}

	// ELF32 and ELF64 loader
	template <int W>
	void Memory<W>::binary_loader(const MachineOptions<W>& options)
	{
		if (UNLIKELY(m_binary.size() < sizeof(Ehdr))) {
			throw std::runtime_error("ELF binary too short");
		}
		const auto* elf = (Ehdr*) m_binary.data();
		if (UNLIKELY(!validate_header<Ehdr> (elf))) {
			throw std::runtime_error("Invalid ELF header! Mixup between 32- and 64-bit?");
		}

		// enumerate & load loadable segments
		const auto program_headers = elf->e_phnum;
		if (UNLIKELY(program_headers <= 0)) {
			throw std::runtime_error("ELF with no program-headers");
		}
		if (UNLIKELY(program_headers >= 10)) {
			throw std::runtime_error("ELF with too many program-headers");
		}
		if (UNLIKELY(elf->e_phoff > 0x4000)) {
			throw std::runtime_error("ELF program-headers have bogus offset");
		}
		if (UNLIKELY(elf->e_phoff + program_headers * sizeof(Phdr) > m_binary.size())) {
			throw std::runtime_error("ELF program-headers are outside the binary");
		}

		const auto* phdr = (Phdr*) (m_binary.data() + elf->e_phoff);
		this->m_start_address = elf->e_entry;

		int seg = 0;
		for (const auto* hdr = phdr; hdr < phdr + program_headers; hdr++)
		{
			// Detect overlapping segments
			for (const auto* ph = phdr; ph < hdr; ph++) {
				if (hdr->p_type == PT_LOAD && ph->p_type == PT_LOAD)
				if (ph->p_vaddr < hdr->p_vaddr + hdr->p_filesz &&
					ph->p_vaddr + ph->p_filesz >= hdr->p_vaddr) {
					// Normally we would not care, but no normal ELF
					// has overlapping segments, so treat as bogus.
					throw std::runtime_error("Overlapping ELF segments");
				}
			}

			switch (hdr->p_type)
			{
				case PT_LOAD:
					// loadable program segments
					if (options.load_program) {
						binary_load_ph(options, hdr);
					}
					seg++;
					break;
				case PT_GNU_STACK:
					//printf("GNU_STACK: 0x%X\n", hdr->p_vaddr);
					this->m_stack_address = hdr->p_vaddr; // ??
					break;
				case PT_GNU_RELRO:
					//throw std::runtime_error(
					//	"Dynamically linked ELF binaries are not supported");
					break;
			}
			if (m_program_end < hdr->p_vaddr + hdr->p_memsz)
				this->m_program_end = hdr->p_vaddr + hdr->p_memsz;
		}

		//this->relocate_section(".rela.dyn", ".symtab");

		// Stack down towards program_end, usually 1mb in size
		this->m_stack_address = m_program_end + options.stack_size;

		// the default exit function is simply 'exit'
		this->m_exit_address = resolve_address("exit");

		if (options.verbose_loader) {
		printf("* Entry is at %p\n", (void*) (uintptr_t) this->start_address());
		}
	}

	template <int W>
	void Memory<W>::machine_loader(const Machine<W>& master)
	{
		auto& mmem = master.memory;
		if (m_main_memory.size() < mmem.m_main_memory.size())
		{
			/* Prevent all sorts of weird problems */
			throw MachineException(OUT_OF_MEMORY,
				"Destination machine has less memory than master",
				m_main_memory.size());
		}

		// copy the whole memory of the master machine
		const auto len = mmem.m_main_memory.max_length(0x0);
		auto* src = mmem.m_main_memory.unsafe_at(0x0, len);
		auto* dst = m_main_memory.unsafe_at(0x0, len);
		__builtin_memcpy(dst, src, len);

		m_main_memory.robase = mmem.m_main_memory.robase;
		m_main_memory.rwbase = mmem.m_main_memory.rwbase;

		this->set_exit_address(mmem.exit_address());
		// base address, size and PC-relative data pointer for instructions
		this->m_exec_pagedata_base = mmem.m_exec_pagedata_base;
		this->m_exec_pagedata_size = mmem.m_exec_pagedata_size;
		this->machine().cpu.initialize_exec_segs(
			mmem.m_exec_pagedata.get() - m_exec_pagedata_base,
			m_exec_pagedata_base, m_exec_pagedata_base + m_exec_pagedata_size);
#ifdef RISCV_INSTR_CACHE
		this->m_exec_decoder = mmem.m_exec_decoder;
#endif
	}

	template <int W>
	const typename Memory<W>::Shdr* Memory<W>::section_by_name(const char* name) const
	{
		const auto* shdr = elf_offset<Shdr> (elf_header()->e_shoff);
		const auto& shstrtab = shdr[elf_header()->e_shstrndx];
		const char* strings = elf_offset<char>(shstrtab.sh_offset);

		for (auto i = 0; i < elf_header()->e_shnum; i++)
		{
			const char* shname = &strings[shdr[i].sh_name];
			if (strcmp(shname, name) == 0) {
				return &shdr[i];
			}
		}
		return nullptr;
	}

	template <int W>
	const typename Elf<W>::Sym* Memory<W>::resolve_symbol(const char* name) const
	{
		if (UNLIKELY(m_binary.empty())) return nullptr;
		const auto* sym_hdr = section_by_name(".symtab");
		if (UNLIKELY(sym_hdr == nullptr)) return nullptr;
		const auto* str_hdr = section_by_name(".strtab");
		if (UNLIKELY(str_hdr == nullptr)) return nullptr;

		const auto* symtab = elf_sym_index(sym_hdr, 0);
		const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf<W>::Sym);
		const char* strtab = elf_offset<char>(str_hdr->sh_offset);

		for (size_t i = 0; i < symtab_ents; i++)
		{
			const char* symname = &strtab[symtab[i].st_name];
			if (strcmp(symname, name) == 0) {
				return &symtab[i];
			}
		}
		return nullptr;
	}


	template <typename Sym>
	static void elf_print_sym(const Sym* sym)
	{
		if constexpr (sizeof(Sym::st_value) == 4) {
			printf("-> Sym is at 0x%X with size %u, type %u name %u\n",
				sym->st_value, sym->st_size,
				ELF32_ST_TYPE(sym->st_info), sym->st_name);
		} else {
			printf("-> Sym is at 0x%lX with size %lu, type %u name %u\n",
				sym->st_value, sym->st_size,
				ELF64_ST_TYPE(sym->st_info), sym->st_name);
		}
	}

	template <int W>
	void Memory<W>::relocate_section(const char* section_name, const char* sym_section)
	{
		const auto* rela = section_by_name(section_name);
		if (rela == nullptr) return;
		const auto* dyn_hdr = section_by_name(sym_section);
		if (dyn_hdr == nullptr) return;
		const size_t rela_ents = rela->sh_size / sizeof(Elf32_Rela);

		auto* rela_addr = elf_offset<Elf32_Rela>(rela->sh_offset);
		for (size_t i = 0; i < rela_ents; i++)
		{
			const uint32_t symidx = ELF32_R_SYM(rela_addr[i].r_info);
			auto* sym = elf_sym_index(dyn_hdr, symidx);

			const uint8_t type = ELF32_ST_TYPE(sym->st_info);
			if (type == STT_FUNC || type == STT_OBJECT)
			{
				auto* entry = elf_offset<address_t> (rela_addr[i].r_offset);
				auto* final = elf_offset<address_t> (sym->st_value);
				if constexpr (true)
				{
					//printf("Relocating rela %zu with sym idx %u where 0x%X -> 0x%X\n",
					//		i, symidx, rela_addr[i].r_offset, sym->st_value);
					elf_print_sym<typename Elf<W>::Sym>(sym);
				}
				*(address_t*) entry = (address_t) (uintptr_t) final;
			}
		}
	}

	template <int W>
	typename Memory<W>::Callsite Memory<W>::lookup(address_t address) const
	{
		const auto* sym_hdr = section_by_name(".symtab");
		if (sym_hdr == nullptr) return {};
		const auto* str_hdr = section_by_name(".strtab");
		if (str_hdr == nullptr) return {};
		// backtrace can sometimes find null addresses
		if (address == 0x0) return {};

		const auto* symtab = elf_sym_index(sym_hdr, 0);
		const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf<W>::Sym);
		const char* strtab = elf_offset<char>(str_hdr->sh_offset);

		const auto result =
			[] (const char* strtab, address_t addr, const auto* sym)
		{
			const char* symname = &strtab[sym->st_name];
			char* dma = __cxa_demangle(symname, nullptr, nullptr, nullptr);
			return Callsite {
				.name = (dma) ? dma : symname,
				.address = sym->st_value,
				.offset = (uint32_t) (addr - sym->st_value),
				.size   = sym->st_size
			};
		};

		const typename Elf<W>::Sym* best = nullptr;
		for (size_t i = 0; i < symtab_ents; i++)
		{
			if (ELF32_ST_TYPE(symtab[i].st_info) != STT_FUNC) continue;
			/*printf("Testing %#X vs  %#X to %#X = %s\n",
					address, symtab[i].st_value,
					symtab[i].st_value + symtab[i].st_size, symname);*/

			if (address >= symtab[i].st_value &&
				address < symtab[i].st_value + symtab[i].st_size)
			{
				// exact match
				return result(strtab, address, &symtab[i]);
			}
			else if (address > symtab[i].st_value)
			{
				// best guess (symbol + 0xOff)
				best = &symtab[i];
			}
		}
		if (best)
			return result(strtab, address, best);
		return {};
	}
	template <int W>
	void Memory<W>::print_backtrace(void(*print_function)(const char*, size_t))
	{
		auto print_trace =
			[this, print_function] (const int N, const address_type<W> addr) {
				// get information about the callsite
				const auto site = this->lookup(addr);
				// write information directly to stdout
				char buffer[8192];
				int len;
				if constexpr (W == 4) {
					len = snprintf(buffer, sizeof(buffer),
						"[%d] 0x%08x + 0x%.3x: %s",
						N, site.address, site.offset, site.name.c_str());
				} else {
					len = snprintf(buffer, sizeof(buffer),
						"[%d] 0x%016lx + 0x%.3x: %s",
						N, site.address, site.offset, site.name.c_str());
				}
				print_function(buffer, len);
			};
		print_trace(0, this->machine().cpu.pc());
		print_trace(1, this->machine().cpu.reg(RISCV::REG_RA));
	}

	template <int W>
	void Memory<W>::protection_fault(address_t addr)
	{
		CPU<W>::trigger_exception(PROTECTION_FAULT, addr);
		__builtin_unreachable();
	}

	template struct Memory<4>;
	template struct Memory<8>;
}
