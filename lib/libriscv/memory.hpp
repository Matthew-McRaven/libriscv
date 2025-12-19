#pragma once
#include <cassert>
#include <cstring>
#include <inttypes.h>
#include <string_view>
#include <unordered_map>
#include "decoded_exec_segment.hpp"
#include "elf.hpp"
#include "mmap_cache.hpp"
#include "page.hpp"
#include "util/buffer.hpp" // <string>
#include "util/function.hpp"
#if RISCV_SPAN_AVAILABLE
#include <span>
#endif
#ifdef RISCV_EXT_ATOMICS
#include "rva.hpp"
#endif

#ifdef __linux__
#include <sys/mman.h>
static constexpr bool MADVISE_ENABLED = true;
#else
extern "C" int madvise(void *, size_t, int);
static constexpr bool MADVISE_ENABLED = false;
#endif
#if defined(__linux__) || defined(__FreeBSD__) || defined(__wasm__)
#define DEMANGLE_ENABLED
#include <sys/mman.h>
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif
extern "C" char *__cxa_demangle(const char *name, char *buf, size_t *n, int *status);
#endif

namespace riscv
{
	template<int W> struct Machine;
	struct vBuffer { char* ptr; size_t len; };

	template<int W>
	struct alignas(RISCV_MACHINE_ALIGNMENT) Memory
	{
		using address_t = address_type<W>;
		using mmio_cb_t = Page::mmio_cb_t;
		using page_fault_cb_t = riscv::Function<Page&(Memory&, address_t, bool)>;
		using page_readf_cb_t = riscv::Function<const Page&(const Memory&, address_t)>;
		using page_write_cb_t = riscv::Function<void(Memory&, address_t, Page&)>;
		static constexpr address_t BRK_MAX      = RISCV_BRK_MEMORY_SIZE; // Default BRK size
		static constexpr address_t DYLINK_BASE  = 0x40000; // Dynamic link base address
		static constexpr address_t RWREAD_BEGIN = 0x1000; // Default rw-arena rodata start

		template <typename T>
		T read(address_t src);

		template <typename T>
		T& writable_read(address_t src);

		template <typename T>
		void write(address_t dst, T value);

		template <typename T>
		void write_paging(address_t dst, T value);

		void memset(address_t dst, uint8_t value, size_t len);
		void memcpy(address_t dst, const void* src, size_t);
		void memcpy(address_t dst, Machine<W>& srcm, address_t src, address_t len);
		void memcpy_out(void* dst, address_t src, size_t) const;
		// Copy between overlapping memory regions. Only available when flat_readwrite_arena is enabled!
		bool try_memmove(address_t dst, address_t src, size_t len);
		// Compare bounded memory
		int memcmp(address_t p1, address_t p2, size_t len) const;
		int memcmp(const void* p1, address_t p2, size_t len) const;
		// Perform the equivalent of MADV_DONTNEED on memory region
		void memdiscard(address_t dst, size_t len, bool ignore_protections);
		/* Fill an array of buffers pointing to complete guest virtual [addr, len].
		   Throws an exception if there was a protection violation.
		   Returns the number of buffers filled, or an exception if not enough. */
		size_t gather_buffers_from_range(size_t cnt, vBuffer[], address_t addr, size_t len) const;
		size_t gather_writable_buffers_from_range(size_t cnt, vBuffer[], address_t addr, size_t len);
		// Gather fragmented virtual memory into a buffer abstraction that can output
		// to a vector, a string and check sequentiality.
		riscv::Buffer membuffer(address_t addr, size_t len, size_t maxlen = 16ul << 20) const;

		/// @brief View known-sequential read/write virtual memory (or throw exception)
		/// @param addr The address to start reading from
		/// @param len The number of bytes to read
		/// @param maxlen The maximum number of bytes to read
		/// @return A string_view of the readable memory
		/// @note Do not attempt to write to the memory region, as it may be read-only
		std::string_view memview(address_t addr, size_t len, size_t maxlen = 16ul << 20) const;

		/// @brief View known-sequential writable virtual memory (or throw exception)
		/// @param addr The address to start reading from
		/// @param len The number of bytes to read
		/// @param maxlen The maximum number of bytes to read
		/// @return A string_view of the readable/writable memory
		std::string_view writable_memview(address_t addr, size_t len, size_t maxlen = 16ul << 20) const;

#ifdef RISCV_SPAN_AVAILABLE
		// View known-sequential virtual memory as array of T with given number of elements (or throw exception)
		// This always uses writable_memview() under the hood to ensure the memory is writable
		template <typename T>
		std::span<T> memspan(address_t addr, size_t elements, size_t maxlen = 16ul << 20) const;
		template <typename T, size_t N>
		std::span<T, N> memspan(address_t addr, size_t maxlen = 16ul << 20) const { return memspan<T>(addr, N, maxlen).template first<N>(); }
#endif

		/// @brief View a fixed-size array of T from a known-sequential virtual memory region
		/// @tparam T The type of the array elements. If T is const, the memory is read-only.
		/// @param addr The address to start reading from
		/// @return A pointer to the array, or nullptr if the memory region is not sequential or too large
		template <typename T, size_t N>
		std::array<T, N>* memarray(address_t addr) const;

		/// @brief View a dynamically-sized array of T from a known-sequential virtual memory region
		/// @tparam T The type of the array elements. If T is const, the memory is read-only.
		/// @param addr The address to start reading from
		/// @param len The number of elements to read
		/// @param maxbytes The maximum number of bytes to read
		/// @return A pointer to the array, or nullptr if the memory region is not sequential or too large
		template <typename T>
		T* memarray(address_t addr, size_t len, size_t maxbytes = 16ul << 20) const;

		// Try viewing a dynamically-sized array of T from a known-sequential virtual memory region
		// Returns nullptr if the memory region is not sequential or too large
		template <typename T>
		T* try_memarray(address_t addr, size_t len, size_t maxbytes = 16ul << 20) const;
		// Read zero-terminated string directly from guests memory, into an owning std::string
		std::string memstring(address_t addr, size_t maxlen = 16384) const;
		size_t strlen(address_t addr, size_t maxlen = 16384) const;
		// Read zero-terminated string directly from guests memory, returning a non-owning string_view
		std::string_view memstring_view(address_t addr, size_t maxlen = 16384) const;

		// Returns the ELF entry/start address (the first instruction)
		address_t start_address() const noexcept { return this->m_start_address; }
		// Returns the current initial stack pointer (unrelated to SP register)
		address_t stack_initial() const noexcept { return this->m_stack_address; }
		void set_stack_initial(address_t addr) { this->m_stack_address = addr; }
		// Returns the address used for exiting (returning from) a vmcall()
		address_t exit_address() const noexcept;
		void      set_exit_address(address_t new_exit);
		// The initial heap address (*not* the current heap maximum)
		address_t heap_address() const noexcept { return this->m_heap_address; }
		// Simple memory mapping implementation
		auto& mmap_cache() noexcept { return m_mmap_cache; }
		address_t mmap_start() const noexcept { return this->m_heap_address + BRK_MAX; }
		const address_t& mmap_address() const noexcept { return m_mmap_address; }
		address_t& mmap_address() noexcept { return m_mmap_address; }
		// Allocate at least writable bytes through mmap(), and return the page-aligned address
		address_t mmap_allocate(address_t bytes);
		// Attempts to relax a previous call to mmap_allocate(), freeing space at the end
		bool mmap_relax(address_t addr, address_t size, address_t new_size);
		// Unmap a memory range
		bool mmap_unmap(address_t addr, address_t size);


		Machine<W>& machine() noexcept { return this->m_machine; }
		const Machine<W>& machine() const noexcept { return this->m_machine; }
		bool is_forked() const noexcept { return !this->m_original_machine; }

#ifdef RISCV_EXT_ATOMICS
		auto& atomics() noexcept { return this->m_atomics; }
		const auto& atomics() const noexcept { return this->m_atomics; }
#endif // RISCV_EXT_ATOMICS

		// Symbol table and section lookup functions
		address_t resolve_address(std::string_view sym) const;
		address_t resolve_section(const char* name) const;
		// Basic backtraces and symbol lookups
		struct Callsite {
			std::string name = "(null)";
			address_t   address = 0x0;
			uint32_t    offset  = 0x0;
			size_t      size    = 0;
		};
		Callsite lookup(address_t) const;
		void print_backtrace(std::function<void(std::string_view)>, bool ra = true) const;

		// Get list of all symbols in the binary
		std::vector<const char*> all_symbols() const;
		// Get list of all unmangled symbols in the binary that starts with a given prefix
		// If the prefix is empty, all unmangled functions are returned
		std::vector<std::string_view> all_unmangled_function_symbols(const std::string& prefix = "") const;
		// Get list of all comments in the ELF binary
		std::vector<std::string_view> elf_comments() const;

		// Counts all the memory used by the machine, execute segments, pages, etc.
		uint64_t memory_usage_total() const noexcept;
		// Helpers for memory usage
		size_t pages_active() const noexcept { return m_pages.size(); }
		size_t owned_pages_active() const noexcept;
		// Page handling
		const auto& pages() const noexcept { return m_pages; }
		auto& pages() noexcept { return m_pages; }
		const Page& get_page(address_t) const;
		const Page& get_exec_pageno(address_t npage) const; // throws
		const Page& get_pageno(address_t npage) const;
		const Page& get_readable_pageno(address_t npage) const;
		Page& create_writable_pageno(address_t npage, bool initialize = true);
		void  set_page_attr(address_t, size_t len, PageAttributes);
		void set_pageno_attr(address_t pageno, PageAttributes);
		std::string get_page_info(address_t addr) const;
		static inline address_t page_number(const address_t address) noexcept {
			return address / Page::size();
		}
		CachedPage<W, const PageData>& rdcache() const noexcept {
			return m_rd_cache;
		}
		CachedPage<W, PageData>& wrcache() noexcept {
			return m_wr_cache;
		}
		const PageData& cached_readable_page(address_t, size_t) const;
		PageData& cached_writable_page(address_t);
		// Page creation & destruction
		template <typename... Args>
		Page& allocate_page(address_t page, Args&& ...);
		void  invalidate_cache(address_t pageno, Page*) const noexcept;
		void  invalidate_reset_cache() const noexcept;
		void  free_pages(address_t, size_t len);
		bool  free_pageno(address_t pageno);

		// Event for writing to unused/unknown memory
		// The old handler is returned, so it can be restored later.
		page_fault_cb_t set_page_fault_handler(page_fault_cb_t h) {
			auto old_handler = std::move(m_page_fault_handler);
			this->m_page_fault_handler = h;
			return old_handler;
		}

		// Event for reading unused/unknown memory
		// The old handler is returned, so it can be restored later.
		page_readf_cb_t set_page_readf_handler(page_readf_cb_t h) {
			auto old_handler = std::move(m_page_readf_handler);
			this->m_page_readf_handler = h;
			return old_handler;
		}
		void reset_page_readf_handler() { this->m_page_readf_handler = default_page_read; }

		// Event for writes on copy-on-write pages
		void set_page_write_handler(page_write_cb_t h) { this->m_page_write_handler = h; }
		static void default_page_write(Memory&, address_t, Page& page);
		static const Page& default_page_read(const Memory&, address_t);
		// NOTE: use print_and_pause() to immediately break!
		void trap(address_t page_addr, mmio_cb_t callback);
		// shared pages (regular pages will have priority!)
		Page&  install_shared_page(address_t pageno, const Page&);
		// create pages for non-owned (shared) memory with given attributes
		void insert_non_owned_memory(
			address_t dst, void* src, size_t size, PageAttributes = {});

		// Custom execute segment, returns page base, final size and execute segment pointer
		std::shared_ptr<DecodedExecuteSegment<W>>& exec_segment_for(address_t vaddr);
		const std::shared_ptr<DecodedExecuteSegment<W>>& exec_segment_for(address_t vaddr) const;
		DecodedExecuteSegment<W>& create_execute_segment(const MachineOptions<W>&, const void* data, address_t addr, size_t len, bool is_initial, bool is_likely_jit = false);
		size_t execute_segments_count() const noexcept { return m_exec.size(); }
		// Evict all execute segments, also disabling the main execute segment
		void evict_execute_segments();
		void evict_execute_segment(DecodedExecuteSegment<W>&);

		const auto& binary() const noexcept { return m_binary; }
		void reset();
		bool is_dynamic_executable() const noexcept { return this->m_is_dynamic; }
		address_t elf_base_address(address_t offset) const;

		bool uses_flat_memory_arena() const noexcept { return riscv::flat_readwrite_arena && this->m_arena.data != nullptr; }
    bool uses_Nbit_encompassing_arena() const noexcept { return false; }
    void *memory_arena_ptr() const noexcept { return (void *)this->m_arena.data; }
    auto &memory_arena_ptr_ref() const noexcept { return this->m_arena.data; }
    size_t memory_arena_size() const noexcept { return this->m_arena.pages * Page::size(); }
    address_t memory_arena_read_boundary() const noexcept { return this->m_arena.read_boundary; }
    address_t memory_arena_write_boundary() const noexcept { return this->m_arena.write_boundary; }
    address_t initial_rodata_end() const noexcept { return this->m_arena.initial_rodata_end; }

    // Serializes the current memory state to an existing vector
    // Returns the final size of the serialized state
    size_t serialize_to(std::vector<uint8_t> &vec) const;
    // Returns memory to a previously stored state
		void deserialize_from(const std::vector<uint8_t>&, const SerializedMachine<W>&);

		Memory(Machine<W>&, std::string_view, MachineOptions<W>);
		Memory(Machine<W>&, const Machine<W>&, MachineOptions<W>);
		~Memory();
	private:
		void clear_all_pages();
		void initial_paging();
		[[noreturn]] static void protection_fault(address_t);
		// Helpers
		template <typename T>
		static void foreach_helper(T& mem, address_t addr, size_t len,
			std::function<void(T&, address_t, const uint8_t*, size_t)> callback);
		template <typename T>
		static void memview_helper(T& mem, address_t addr, size_t len,
			std::function<void(T&, const uint8_t*, size_t)> callback);
		// ELF stuff
		using Elf = typename riscv::Elf<W>;
		template <typename T> T* elf_offset(size_t ofs) const {
			if (ofs + sizeof(T) >= ofs && ofs + sizeof(T) < m_binary.size())
				return (T*) &m_binary[ofs];
#if __cpp_exceptions
			throw MachineException(INVALID_PROGRAM, "Invalid ELF offset", ofs);
#else
			std::abort();
#endif

		}
		const auto* elf_header() const {
			return elf_offset<const typename Elf::Header> (0);
		}
		const typename Elf::SectionHeader* section_by_name(const std::string& name) const;
		void dynamic_linking(const typename Elf::Header&);
		void relocate_section(const char* section_name, const char* symtab);
		const typename Elf::Sym* resolve_symbol(std::string_view name) const;
		const typename Elf::Sym* elf_sym_index(const typename Elf::SectionHeader* shdr, uint32_t symidx) const;
		// ELF loader
		void binary_loader(const MachineOptions<W>&);
		void binary_load_ph(const MachineOptions<W>&, const typename Elf::ProgramHeader*, address_t vaddr);
		void serialize_execute_segment(const MachineOptions<W>&, const typename Elf::ProgramHeader*, address_t vaddr);
		void generate_decoder_cache(const MachineOptions<W>&, std::shared_ptr<DecodedExecuteSegment<W>>&, bool is_initial);
		// Machine copy-on-write fork
		void machine_loader(const Machine<W>&, const MachineOptions<W>&);

		address_t m_start_address = 0;
		address_t m_stack_address = 0;
		address_t m_exit_address  = 0;
		address_t m_mmap_address  = 0;
		address_t m_heap_address  = 0;

		Machine<W>& m_machine;

		mutable CachedPage<W, const PageData> m_rd_cache;
		mutable CachedPage<W, PageData> m_wr_cache;

		std::unordered_map<address_t, Page> m_pages;

		const bool m_original_machine;
		bool m_is_dynamic = false;

		const std::string_view m_binary;

		// Memory map cache
		MMapCache<W> m_mmap_cache;

		page_fault_cb_t m_page_fault_handler = nullptr;
		page_write_cb_t m_page_write_handler = default_page_write;
		page_readf_cb_t m_page_readf_handler = default_page_read;

#ifdef RISCV_EXT_ATOMICS
		AtomicMemory<W> m_atomics;
#endif

		// Execute segments
		std::shared_ptr<DecodedExecuteSegment<W>> m_main_exec_segment;
		std::vector<std::shared_ptr<DecodedExecuteSegment<W>>> m_exec; // not including main_exec_segment
		std::shared_ptr<DecodedExecuteSegment<W>>& next_execute_segment();

		// Linear arena at start of memory (mmap-backed)
		struct {
			PageData* data = nullptr;
			address_t read_boundary = 0;
			address_t write_boundary = 0;
			address_t initial_rodata_end = 0;
			size_t    pages = 0;
		} m_arena;

		friend struct CPU<W>;
	};
#include "memory_helpers_paging.hpp"
#include "memory_inline.hpp"
#include "memory_inline_pages.hpp"
  // memory.cpp
  template <int W>
  Memory<W>::Memory(Machine<W> &mach, std::string_view bin, MachineOptions<W> options)
      : m_machine{mach}, m_original_machine{true}, m_binary{bin} {
    if (options.page_fault_handler != nullptr) {
      this->m_page_fault_handler = std::move(options.page_fault_handler);
    } else if (options.memory_max != 0) {
      const address_t pages_max = options.memory_max / Page::size();
      assert(pages_max >= 1);

      if (options.use_memory_arena) {
#if defined(__linux__) || defined(__FreeBSD__)

        // Over-allocate by 1 page in order to avoid bounds-checking with size
        const size_t len = (pages_max + 1) * Page::size();
        this->m_arena.data =
            (PageData *)mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
        this->m_arena.pages = pages_max;
        // mmap() returns MAP_FAILED (-1) when mapping fails
        if (UNLIKELY(this->m_arena.data == MAP_FAILED)) {
          this->m_arena.data = nullptr;
          this->m_arena.pages = 0;
        }
#else
        // TODO: XXX: Investigate if this is a time sink
        this->m_arena.data = new PageData[pages_max + 1];
        this->m_arena.pages = pages_max;
#endif
      }

      if (this->m_arena.pages > 0) {
        // There is now a sequential arena, but we should make room for
        // some pages that can appear anywhere in the address space.
        const unsigned anywhere_pages = pages_max / 2;
        this->m_page_fault_handler = [anywhere_pages](auto &mem, const address_t page, bool init) -> Page & {
          if (mem.pages_active() < anywhere_pages || mem.owned_pages_active() < anywhere_pages) {
            // Within linear arena at the start
            if (page < mem.m_arena.pages) {
              const PageAttributes attr{.read = true, .write = true, .non_owning = true};
              return mem.allocate_page(page, attr, &mem.m_arena.data[page]);
            }
            // Create page on-demand
            return mem.allocate_page(page, init ? PageData::INITIALIZED : PageData::UNINITIALIZED);
          }
          // Out of memory, which is (2 + 1) * anywhere_pages
          throw MachineException(OUT_OF_MEMORY, "Out of memory", anywhere_pages * 3);
        };
      } else {
        this->m_page_fault_handler = [pages_max](auto &mem, const address_t page, bool init) -> Page & {
          if (mem.pages_active() < pages_max || mem.owned_pages_active() < pages_max) {
            // Create page on-demand
            return mem.allocate_page(page, init ? PageData::INITIALIZED : PageData::UNINITIALIZED);
          }
          throw MachineException(OUT_OF_MEMORY, "Out of memory", pages_max);
        };
      }
    } else {
      throw MachineException(OUT_OF_MEMORY, "Max memory was zero", 0);
    }
    if (!m_binary.empty()) {
      // Add a zero-page at the start of address space
      this->initial_paging();
      // load ELF binary into virtual memory
      this->binary_loader(options);
    }
  }
  template <int W>
  Memory<W>::Memory(Machine<W> &mach, const Machine<W> &other, MachineOptions<W> options)
      : m_machine{mach}, m_original_machine{false}, m_binary{other.memory.binary()} {
#ifdef RISCV_EXT_ATOMICS
    this->m_atomics = other.memory.m_atomics;
#endif
    this->machine_loader(other, options);
  }

  template <int W> Memory<W>::~Memory() {
    try {
      this->clear_all_pages();
    } catch (...) {
    }
    // Potentially deallocate execute segments that are no longer referenced
    this->evict_execute_segments();
    // only the original machine owns arena
    if (this->m_arena.data != nullptr && !is_forked()) {
#if defined(__linux__) || defined(__FreeBSD__)
      munmap(this->m_arena.data, (this->m_arena.pages + 1) * Page::size());
#else
      delete[] this->m_arena.data;
#endif
    }
  }

  template <int W> RISCV_INTERNAL void Memory<W>::reset() {
    // Hard to support because of things like
    // serialization, machine options and machine forks
  }

  template <int W> void Memory<W>::clear_all_pages() {
    this->m_pages.clear();
    this->invalidate_reset_cache();
  }

  template <int W> RISCV_INTERNAL void Memory<W>::initial_paging() {
    if (m_pages.find(0) == m_pages.end()) {
      // add a guard page to catch zero-page accesses
      install_shared_page(0, Page::guard_page());
    }
  }

  template <int W>
  RISCV_INTERNAL void Memory<W>::binary_load_ph(const MachineOptions<W> &options,
                                                const typename Elf::ProgramHeader *hdr, const address_t vaddr) {
    const auto *src = m_binary.data() + hdr->p_offset;
    const size_t len = hdr->p_filesz;
    if (m_binary.size() <= hdr->p_offset || hdr->p_offset + len < hdr->p_offset) {
      throw MachineException(INVALID_PROGRAM, "Bogus ELF program segment offset");
    }
    if (m_binary.size() < hdr->p_offset + len) {
      throw MachineException(INVALID_PROGRAM, "Not enough room for ELF program segment");
    }
    if (vaddr + len < vaddr) {
      throw MachineException(INVALID_PROGRAM, "Bogus ELF segment virtual base");
    }

    if (options.verbose_loader) {
      printf("* Loading program of size %zu from %p to virtual %p -> %p\n", len, src, (void *)uintptr_t(vaddr),
             (void *)uintptr_t(vaddr + len));
    }
    // Serialize pages cannot be called with len == 0,
    // and there is nothing further to do.
    if (UNLIKELY(len == 0)) return;

    // segment permissions
    const PageAttributes attr{.read = (hdr->p_flags & Elf::PF_R) != 0,
                              .write = (hdr->p_flags & Elf::PF_W) != 0,
                              .exec = (hdr->p_flags & Elf::PF_X) != 0};
    if (options.verbose_loader) {
      printf("* Program segment readable: %d writable: %d  executable: %d\n", attr.read, attr.write, attr.exec);
    }

    if (attr.read && !attr.write && uses_flat_memory_arena()) {
      this->m_arena.initial_rodata_end = std::max(m_arena.initial_rodata_end, static_cast<address_t>(vaddr + len));
    }
    // Nothing more to do here, if execute-only
    if (attr.exec && !attr.read) return;
    // We would normally never allow this
    if (attr.exec && attr.write) {
      if (!options.allow_write_exec_segment) {
        throw MachineException(INVALID_PROGRAM,
                               "Insecure ELF has writable executable code (Disable check in MachineOptions)");
      }
    }
    // In some cases we want to enforce execute-only
    if (attr.exec && (attr.read || attr.write)) {
      if (options.enforce_exec_only) {
        throw MachineException(INVALID_PROGRAM, "Execute segment must be execute-only");
      }
    }

    // Load into virtual memory
    this->memcpy(vaddr, src, len);

    if (options.protect_segments) {
      this->set_page_attr(vaddr, len, attr);
    } else {
      // this might help execute simplistic barebones programs
      this->set_page_attr(vaddr, len, {.read = true, .write = true, .exec = true});
    }
  }

  template <int W>
  RISCV_INTERNAL void Memory<W>::serialize_execute_segment(const MachineOptions<W> &options,
                                                           const typename Elf::ProgramHeader *hdr, address_t vaddr) {
    // The execute segment:
    size_t exlen = hdr->p_filesz;
    const char *data = m_binary.data() + hdr->p_offset;

    // Zig's ELF writer is insane, so we add an option to disable .text section segment reduction.
    if (W <= 8 && !options.ignore_text_section) {
      // Look for a .text section inside this segment:
      const auto *texthdr = section_by_name(".text");
      if (texthdr != nullptr
          // Validate that the .text section is inside this
          // execute segment.
          && texthdr->sh_addr >= vaddr && texthdr->sh_size <= exlen &&
          texthdr->sh_addr + texthdr->sh_size <= vaddr + exlen) {
        data = m_binary.data() + texthdr->sh_offset;
        vaddr = this->elf_base_address(texthdr->sh_addr);
        exlen = texthdr->sh_size;
        // Work-around for Zig's __lcxx_override section
        // It comes right after .text, so we can merge them
        // TODO: Automatically merge sections that are adjacent
        const auto *lcxxhdr = section_by_name("__lcxx_override");
        if (lcxxhdr != nullptr && lcxxhdr->sh_addr == texthdr->sh_addr + texthdr->sh_size) {
          const unsigned size = texthdr->sh_size + lcxxhdr->sh_size;
          if (size <= hdr->p_filesz && texthdr->sh_addr + size <= vaddr + hdr->p_filesz) {
            // Merge the two sections
            exlen = size;
          } else if (options.verbose_loader) {
            printf("* __lcxx_override section is outside of program header: %p -> %p where %zu <= %zu\n",
                   (void *)uintptr_t(vaddr), (void *)uintptr_t(vaddr + exlen), size_t(size), size_t(hdr->p_filesz));
          }
        }
      }
      // printf("* Found .text section inside segment: %p -> %p\n",
      //	(void*)uintptr_t(vaddr), (void*)uintptr_t(vaddr + exlen));
    }

    // Create an *initial* execute segment
    auto &exec_segment = this->create_execute_segment(options, data, vaddr, exlen, true);
    // Set the segment as execute-only when R|W are not set
    exec_segment.set_execute_only((hdr->p_flags & (Elf::PF_R | Elf::PF_W)) == 0);
    // Select the first execute segment
    if (machine().cpu.current_execute_segment().empty()) machine().cpu.set_execute_segment(exec_segment);
  }

  // ELF32 and ELF64 loader
  template <int W> RISCV_INTERNAL void Memory<W>::binary_loader(const MachineOptions<W> &options) {
    static constexpr uint32_t ELFHDR_FLAGS_RVC = 0x1;
    static constexpr uint32_t ELFHDR_FLAGS_RVE = 0x8;

    if (UNLIKELY(m_binary.size() < sizeof(typename Elf::Header))) {
      throw MachineException(INVALID_PROGRAM, "ELF program too short");
    }
    if (UNLIKELY(!Elf::validate(m_binary))) {
      if constexpr (W == 4)
        throw MachineException(INVALID_PROGRAM, "Invalid ELF header! Expected a 32-bit RISC-V ELF binary");
      else if constexpr (W == 8)
        throw MachineException(INVALID_PROGRAM, "Invalid ELF header! Expected a 64-bit RISC-V ELF binary");
      else if constexpr (W == 16)
        throw MachineException(INVALID_PROGRAM, "Invalid ELF header! Expected a 128-bit RISC-V ELF binary");
      else throw MachineException(INVALID_PROGRAM, "Invalid ELF header! Expected a RISC-V ELF binary");
    }

    const auto *elf = (typename Elf::Header *)m_binary.data();
    const bool is_static = elf->e_type == Elf::Header::ET_EXEC;
    this->m_is_dynamic = elf->e_type == Elf::Header::ET_DYN;
    if (UNLIKELY(!is_static && !m_is_dynamic)) {
      throw MachineException(INVALID_PROGRAM, "ELF program is not an executable type. Trying to load an object file?");
    }
    if (UNLIKELY(elf->e_machine != Elf::Header::EM_RISCV)) {
      throw MachineException(INVALID_PROGRAM, "ELF program is not a RISC-V executable. Wrong architecture.");
    }
    if (UNLIKELY((elf->e_flags & ELFHDR_FLAGS_RVC) != 0 && !compressed_enabled)) {
      throw MachineException(INVALID_PROGRAM, "ELF is a RISC-V RVC executable, however C-extension is not enabled.");
    }
    if (UNLIKELY((elf->e_flags & ELFHDR_FLAGS_RVE) != 0)) {
      throw MachineException(INVALID_PROGRAM, "ELF is a RISC-V RVE executable, however E-extension is not supported.");
    }

    // Enumerate & validate loadable segments
    const auto program_headers = elf->e_phnum;
    if (UNLIKELY(program_headers <= 0)) {
      throw MachineException(INVALID_PROGRAM, "ELF with no program-headers");
    }
    if (UNLIKELY(program_headers >= 16)) {
      throw MachineException(INVALID_PROGRAM, "ELF with too many program-headers");
    }
    if (UNLIKELY(elf->e_phoff > 0x4000)) {
      throw MachineException(INVALID_PROGRAM, "ELF program-headers have bogus offset");
    }
    if (UNLIKELY(elf->e_phoff + program_headers * sizeof(typename Elf::ProgramHeader) > m_binary.size())) {
      throw MachineException(INVALID_PROGRAM, "ELF program-headers are outside the binary");
    }

    // Load program segments
    const auto *phdr = (typename Elf::ProgramHeader *)(m_binary.data() + elf->e_phoff);
    std::vector<const typename Elf::ProgramHeader *> execute_segments;

    // is_dynamic() is used to determine the ELF base address
    this->m_start_address = this->elf_base_address(elf->e_entry);
    this->m_heap_address = 0;

    for (const auto *hdr = phdr; hdr < phdr + program_headers; hdr++) {
      const address_t vaddr = this->elf_base_address(hdr->p_vaddr);

      // Detect overlapping segments
      for (const auto *ph = phdr; ph < hdr; ph++) {
        const address_t ph_vaddr = this->elf_base_address(ph->p_vaddr);

        if (hdr->p_type == Elf::PT_LOAD && ph->p_type == Elf::PT_LOAD)
          if (ph_vaddr < vaddr + hdr->p_filesz && ph_vaddr + ph->p_filesz > vaddr) {
            // Normally we would not care, but no normal ELF
            // has overlapping segments, so treat as bogus.
            throw MachineException(INVALID_PROGRAM, "Overlapping ELF segments");
          }
      }

      switch (hdr->p_type) {
      case Elf::PT_LOAD:
        // loadable program segments
        if (options.load_program) {
          binary_load_ph(options, hdr, vaddr);
          if (hdr->p_flags & Elf::PF_X) {
            execute_segments.push_back(hdr);
          }
        }
        break;
      case Elf::PT_GNU_STACK:
        // This seems to be a mark for executable stack. Big NO!
        break;
      case Elf::PT_GNU_RELRO:
        /*this->set_page_attr(vaddr, hdr->p_memsz, {
          .read  = (hdr->p_flags & PF_R) != 0,
          .write = (hdr->p_flags & PF_W) != 0,
          .exec  = (hdr->p_flags & PF_X) != 0,
        });*/
        break;
      }

      address_t endm = vaddr + hdr->p_memsz;
      endm += Page::size() - 1;
      endm &= ~address_t(Page::size() - 1);
      if (this->m_heap_address < endm) this->m_heap_address = endm;
    }

    // The base mmap address starts at heap start + BRK_MAX
    // TODO: We should check if the heap starts too close to the end
    // of the address space now, and move it around if necessary.
    this->m_mmap_address = m_heap_address + BRK_MAX;

    // Default stack
    this->m_stack_address = mmap_allocate(options.stack_size) + options.stack_size;

    if (!options.default_exit_function.empty()) {
      // It is slightly faster to set a custom exit function, in order
      // to avoid changing execute segment (slow-path) to exit.
      auto potential_exit_addr = this->resolve_address(options.default_exit_function);
      if (potential_exit_addr != 0x0) {
        this->m_exit_address = potential_exit_addr;
        if (UNLIKELY(options.verbose_loader)) {
          printf("* Using program-provided exit function at %p\n", (void *)uintptr_t(this->exit_address()));
        }
      }
    }
    // Default fallback: Install our own exit function as a separate execute segment
    if (this->m_exit_address == 0x0) {
      // Insert host code page, with exit function, enabling VM calls.
      auto host_page = this->mmap_allocate(Page::size());
      this->install_shared_page(page_number(host_page), Page::host_page());
      this->m_exit_address = host_page;
    }

    if (this->uses_flat_memory_arena() && this->memory_arena_size() >= m_arena.initial_rodata_end) {
      this->m_arena.read_boundary =
          std::min(this->memory_arena_size(), size_t(this->memory_arena_size() - RWREAD_BEGIN));
      this->m_arena.write_boundary =
          std::min(this->memory_arena_size(), size_t(this->memory_arena_size() - m_arena.initial_rodata_end));
    } else {
      this->m_arena.initial_rodata_end = 0;
    }

    // Now that we know the boundries of the program, generate
    // efficient execute segments (if loadable).
    if (options.load_program) {
      for (auto *hdr : execute_segments) {
        const address_t vaddr = this->elf_base_address(hdr->p_vaddr);

        serialize_execute_segment(options, hdr, vaddr);
      }
      if constexpr (W <= 8) {
        if (this->m_is_dynamic) {
          this->dynamic_linking(*elf);
        }
      }
    }

    if (UNLIKELY(options.verbose_loader)) {
      printf("* Entry is at %p\n", (void *)uintptr_t(this->start_address()));
    }
  }

  template <int W>
  RISCV_INTERNAL void Memory<W>::machine_loader(const Machine<W> &master, const MachineOptions<W> &options) {
    // Some machines don't need custom PF handlers
    this->m_page_fault_handler = master.memory.m_page_fault_handler;

    if (options.minimal_fork == false) {
      // Hardly any pages are dont_fork, so we estimate that
      // all master pages will be loaned.
      m_pages.reserve(master.memory.pages().size());

      for (const auto &it : master.memory.pages()) {
        const auto &page = it.second;
        // Skip pages marked as dont_fork
        if (page.attr.dont_fork) continue;
        // Make every page non-owning
        auto attr = page.attr;
        if (attr.write) {
          attr.write = false;
          attr.is_cow = true;
        }
        attr.non_owning = true;
        m_pages.try_emplace(it.first, attr, page.m_page.get());
      }
    }
    this->m_start_address = master.memory.m_start_address;
    this->m_stack_address = master.memory.m_stack_address;
    this->m_exit_address = master.memory.m_exit_address;
    this->m_heap_address = master.memory.m_heap_address;
    this->m_mmap_address = master.memory.m_mmap_address;
    this->m_mmap_cache = master.memory.m_mmap_cache;

    // Reference the same execute segments
    this->m_main_exec_segment = master.memory.m_main_exec_segment;
    this->m_exec = master.memory.m_exec;

    if (options.use_memory_arena) {
      this->m_arena.data = master.memory.m_arena.data;
      this->m_arena.pages = master.memory.m_arena.pages;
      this->m_arena.read_boundary = master.memory.m_arena.read_boundary;
      this->m_arena.write_boundary = master.memory.m_arena.write_boundary;
      this->m_arena.initial_rodata_end = master.memory.m_arena.initial_rodata_end;
    }

    // invalidate all cached pages, because references are invalidated
    this->invalidate_reset_cache();
  }

  template <int W> std::string Memory<W>::get_page_info(address_t addr) const {
    char buffer[1024];
    int len;
    if constexpr (W == 4) {
      len = snprintf(buffer, sizeof(buffer), "[0x%08" PRIX32 "] %s", addr, get_page(addr).to_string().c_str());
    } else if constexpr (W == 8) {
      len = snprintf(buffer, sizeof(buffer), "[0x%016" PRIX64 "] %s", addr, get_page(addr).to_string().c_str());
    } else if constexpr (W == 16) {
      len =
          snprintf(buffer, sizeof(buffer), "[0x%016" PRIX64 "] %s", (uint64_t)addr, get_page(addr).to_string().c_str());
    }
    return std::string(buffer, len);
  }

  template <int W> typename Memory<W>::Callsite Memory<W>::lookup(address_t address) const {
    if (!Elf::validate(this->m_binary)) return {};

    const auto *sym_hdr = section_by_name(".symtab");
    if (sym_hdr == nullptr) return {};
    const auto *str_hdr = section_by_name(".strtab");
    if (str_hdr == nullptr) return {};
    // backtrace can sometimes find null addresses
    if (address == 0x0) return {};
    // ELF with no symbols
    if (UNLIKELY(sym_hdr->sh_size == 0)) return {};

    // Add the correct offset to address for dynamically loaded programs
    address = this->elf_base_address(address);

    const auto *symtab = elf_offset<typename Elf::Sym>(sym_hdr->sh_offset);
    const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf::Sym);
    const char *strtab = elf_offset<char>(str_hdr->sh_offset);

    const auto result = [](const char *strtab, address_t addr, const auto *sym) {
      const char *symname = &strtab[sym->st_name];
      std::string result;
#ifdef DEMANGLE_ENABLED
      if (char *dma = __cxa_demangle(symname, nullptr, nullptr, nullptr); dma != nullptr) {
        result = dma;
        free(dma);
      } else {
        result = symname;
      }
#else
      result = symname;
#endif
      return Callsite{.name = result,
                      .address = static_cast<address_t>(sym->st_value),
                      .offset = (uint32_t)(addr - sym->st_value),
                      .size = size_t(sym->st_size)};
    };

    const typename Elf::Sym *best = nullptr;
    for (size_t i = 0; i < symtab_ents; i++) {
      if (Elf::SymbolType(symtab[i].st_info) != Elf::STT_FUNC) continue;
      /*printf("Testing %#X vs  %#X to %#X = %s\n",
          address, symtab[i].st_value,
          symtab[i].st_value + symtab[i].st_size, symname);*/

      if (address >= symtab[i].st_value && address < symtab[i].st_value + symtab[i].st_size) {
        // The current symbol was the best match
        return result(strtab, address, &symtab[i]);
      } else if (address >= symtab[i].st_value && (!best || symtab[i].st_value > best->st_value)) {
        // best guess (symbol + 0xOff)
        best = &symtab[i];
      }
    }
    if (best) return result(strtab, address, best);
    return {};
  }
  template <int W>
  void Memory<W>::print_backtrace(std::function<void(std::string_view)> print_function, bool ra) const {
    auto print_trace = [this, print_function](const int N, const address_type<W> addr) {
      // get information about the callsite
      const auto site = this->lookup(addr);
      if (site.address == 0 && site.offset == 0 && site.size == 0) {
        // if there is nothing to print, indicate that this is
        // an unknown/empty location by "printing" a zero-length string.
        print_function({});
        return;
      }

      // write information directly to stdout
      char buffer[8192];
      int len = 0;
      if (N >= 0) {
        len = snprintf(&buffer[len], sizeof(buffer) - len, "[%d] ", N);
      }
      if constexpr (W == 4) {
        len += snprintf(&buffer[len], sizeof(buffer) - len, "0x%08" PRIx32 " + 0x%.3" PRIx32 ": %s", site.address,
                        site.offset, site.name.c_str());
      } else if constexpr (W == 8) {
        len += snprintf(&buffer[len], sizeof(buffer) - len, "0x%016" PRIX64 " + 0x%.3" PRIx32 ": %s", site.address,
                        site.offset, site.name.c_str());
      } else if constexpr (W == 16) {
        len += snprintf(&buffer[len], sizeof(buffer) - len, "0x%016" PRIx64 " + 0x%.3" PRIx32 ": %s",
                        (uint64_t)site.address, site.offset, site.name.c_str());
      }
      if (len > 0) print_function({buffer, (size_t)len});
      else print_function("Scuffed frame. Should not happen!");
    };
    if (ra) {
      print_trace(0, this->machine().cpu.pc());
      print_trace(1, this->machine().cpu.reg(REG_RA));
    } else {
      print_trace(-1, this->machine().cpu.pc());
    }
  }

  template <int W> void Memory<W>::protection_fault(address_t addr) {
    CPU<W>::trigger_exception(PROTECTION_FAULT, addr);
  }

  // memory_rw.cpp
  template <int W> const Page &Memory<W>::get_readable_pageno(const address_t pageno) const {
    const auto &page = get_pageno(pageno);
    if (LIKELY(page.attr.read)) return page;
    this->protection_fault(pageno * Page::size());
  }

  template <int W> Page &Memory<W>::create_writable_pageno(const address_t pageno, bool init) {
    auto it = m_pages.find(pageno);
    if (LIKELY(it != m_pages.end())) {
      Page &page = it->second;
      if (LIKELY(page.attr.write)) {
        return page;
      } else if (page.attr.is_cow) {
        m_page_write_handler(*this, pageno, page);
        // The page may be read-cached at this time
        // and the page data has likely changed now.
        this->invalidate_cache(pageno, &page);
        return page;
      }
    } else {
      // Handler must produce a new page, or throw
      Page &page = m_page_fault_handler(*this, pageno, init);
      if (LIKELY(page.attr.write)) {
        this->invalidate_cache(pageno, &page);
        return page;
      }
    }
    this->protection_fault(pageno * Page::size());
  }

  template <int W> void Memory<W>::set_pageno_attr(const address_t pageno, PageAttributes attr) {
    auto it = pages().find(pageno);
    if (it != pages().end()) {
      auto &page = it->second;
      // Keep non-owning and is_cow attributes
      const bool is_cow = page.attr.is_cow;
      page.attr.apply_regular_attributes(attr);
      // If the page becomes writable and holds the CoW-page data, it's also copy-on-write
      if (is_cow || (attr.write && page.is_cow_page())) {
        page.attr.is_cow = true;
        page.attr.write = false;
      }
      return;
    }

    // Create arena-page
    if (flat_readwrite_arena && pageno < this->m_arena.pages) {
      auto &page = this->create_writable_pageno(pageno);
      page.attr.apply_regular_attributes(attr);
      return;
    }

    // Don't create any pages if the defaults apply
    const bool is_default = attr.is_default();
    if (is_default) return;

    // Writable: Create a non-owning copy-on-write zero-page
    // Read-only: Create a non-owning zero-page
    // Unmapped: Create hidden non-owning zero-page, which can become copy-on-write
    attr.is_cow = attr.write;
    attr.write = false;
    attr.non_owning = true;
    m_pages.try_emplace(pageno, attr, Page::cow_page().m_page.get());
  }

  template <int W> void Memory<W>::memdiscard(address_t dst, size_t len, bool ignore_protections) {
#ifndef MADV_DONTNEED
    static constexpr int MADV_DONTNEED = 0x4;
#endif
    while (len > 0) {
      const size_t offset = dst & (Page::size() - 1); // offset within page
      const size_t size = std::min(Page::size() - offset, len);
      const address_t pageno = page_number(dst);

      // We only use the page table now because we have previously
      // checked special regions.
      auto it = m_pages.find(pageno);
      // If we don't find a page, we can treat it as a CoW zero page
      if (it != m_pages.end()) {
        Page &page = it->second;
        if (page.is_cow_page()) {
          // This is the zero-page
        } else {
          if (page.attr.is_cow) {
            m_page_write_handler(*this, pageno, page);
          }
          if (page.attr.write || ignore_protections) {

            if constexpr (MADVISE_ENABLED) {
              // madvise "fast-path" (XXX: doesn't scale on busy server)
              if (offset == 0 && size == Page::size()) {
                madvise(page.data(), Page::size(), MADV_DONTNEED);
              } else {
                std::memset(page.data() + offset, 0, size);
              }
            } else {
              // Zero the existing writable page
              std::memset(page.data() + offset, 0, size);
            }

          } else if (!ignore_protections) {
            this->protection_fault(dst);
          }
        }
      } else {
        // Create arena-page
        if (flat_readwrite_arena && pageno < this->m_arena.pages) {
          // Fast-path using madvise
          if constexpr (MADVISE_ENABLED) {
            // XXX: doesn't scale on busy server
            if (offset == 0 && size == Page::size()) {
              address_t new_dst = dst + (len & ~address_t(Page::size() - 1));
              new_dst = std::min(new_dst, (address_t)memory_arena_size());
              const size_t new_size = new_dst - dst;

              auto *baseptr = &((uint8_t *)m_arena.data)[dst];
              madvise(baseptr, Page::size(), MADV_DONTNEED);

              dst += new_size;
              len -= new_size;
              continue;
            }
          }

          auto &page = this->create_writable_pageno(pageno);
          // Unfortunately we don't know if this page is untouched,
          // but we can use MADV_DONTNEED
          if (page.attr.write || ignore_protections) {
            std::memset(page.data() + offset, 0, size);
          } else if (!ignore_protections) {
            this->protection_fault(dst);
          }
        } else {
          // Assume this page is lazily created (zero-page)
        }
      }

      dst += size;
      len -= size;
    }
  }

  template <int W> bool Memory<W>::free_pageno(address_t pageno) { return m_pages.erase(pageno) != 0; }

  template <int W> void Memory<W>::free_pages(address_t dst, size_t len) {
    address_t pageno = page_number(dst);
    address_t end = pageno + page_number((len + (Page::size() - 1)) & ~(Page::size() - 1));
    while (pageno < end) {
      this->free_pageno(pageno);
      pageno++;
    }
    // TODO: This can be improved by invalidating matches only
    this->invalidate_reset_cache();
  }

  template <int W> void Memory<W>::default_page_write(Memory<W> &, address_t, Page &page) { page.make_writable(); }

  template <int W> const Page &Memory<W>::default_page_read(const Memory<W> &mem, address_t pageno) {
    // This is a copy-on-write zeroed area, but we must respect the underlying arena
    if (flat_readwrite_arena && pageno < mem.m_arena.pages) {
      return const_cast<Memory<W> &>(mem).create_writable_pageno(pageno);
    }
    return Page::cow_page();
  }

  template <int W> Page &Memory<W>::install_shared_page(address_t pageno, const Page &shared_page) {
    auto &already_there = get_pageno(pageno);
    if (!already_there.is_cow_page() && !already_there.attr.non_owning)
      throw MachineException(ILLEGAL_OPERATION, "There was a page at the specified location already", pageno);

    auto attr = shared_page.attr;
    attr.non_owning = true;
    // NOTE: If you insert a const Page, DON'T modify it! The machine
    // won't, unless system-calls do or manual intervention happens!
    auto res = m_pages.try_emplace(pageno, attr, const_cast<PageData *>(shared_page.m_page.get()));
    // TODO: Can be improved by invalidating more intelligently
    this->invalidate_reset_cache();
    // try overwriting instead, if emplace failed
    if (res.second == false) {
      Page &page = res.first->second;
      new (&page) Page{attr, const_cast<PageData *>(shared_page.m_page.get())};
      return page;
    }
    return res.first->second;
  }

  template <int W> void Memory<W>::insert_non_owned_memory(address_t dst, void *src, size_t size, PageAttributes attr) {
    assert(dst % Page::size() == 0);
    assert((dst + size) % Page::size() == 0);
    attr.non_owning = true;

    for (size_t i = 0; i < size; i += Page::size()) {
      const auto pageno = (dst + i) / Page::size();
      PageData *pdata = reinterpret_cast<PageData *>((char *)src + i);
      m_pages.try_emplace(pageno, attr, pdata);
    }
    // TODO: Can be improved by invalidating more intelligently
    this->invalidate_reset_cache();
  }

  template <int W> void Memory<W>::set_page_attr(address_t dst, size_t len, PageAttributes attr) {
    // printf("set_page_attr(0x%lX, %zu, prot=%X)\n", long(dst), len, attr.to_prot());
    while (len > 0) {
      const size_t offset = dst & (Page::size() - 1); // offset within page
      const size_t size = std::min(Page::size() - offset, len);
      const address_t pageno = page_number(dst);
      this->set_pageno_attr(pageno, attr);

      dst += size;
      len -= size;
    }
  }

  template <int W> uint64_t Memory<W>::memory_usage_total() const noexcept {
    uint64_t total = 0;
    total += sizeof(Machine<W>);
    // Pages
    for (const auto &it : m_pages) {
      const auto page_number = it.first;
      const auto &page = it.second;
      total += sizeof(page);
      // Regular owned page (that is not the shared zero-page)
      if ((!page.attr.non_owning && !page.is_cow_page()) ||
          // Arena page
          (page.attr.non_owning && page_number < m_arena.pages))
        total += Page::size();
    }

    for (const auto &exec : m_exec) {
      if (exec) total += exec->size_bytes();
    }

    return total;
  }

  // memory_mmap.cpp
  template <int W> address_type<W> Memory<W>::mmap_allocate(address_t bytes) {
    // Bytes rounded up to nearest PageSize.
    const address_t result = this->m_mmap_address;
    this->m_mmap_address += (bytes + PageMask) & ~address_t{PageMask};
    return result;
  }

  template <int W> bool Memory<W>::mmap_relax(address_t addr, address_t size, address_t new_size) {
    // Undo or relax the last mmap allocation. Returns true if successful.
    if (this->m_mmap_address == addr + size && new_size <= size) {
      this->m_mmap_address = (addr + new_size + PageMask) & ~address_t{PageMask};
      return true;
    }
    return false;
  }

  template <int W> bool Memory<W>::mmap_unmap(address_t addr, address_t size) {
    size = (size + PageMask) & ~address_t{PageMask};
    const bool relaxed = this->mmap_relax(addr, size, 0u);
    if (relaxed) {
      // If relaxation happened, invalidate intersecting cache entries.
      this->mmap_cache().invalidate(addr, size);
    } else if (addr >= this->mmap_start()) {
      // If relaxation didn't happen, put in the cache for later.
      this->mmap_cache().insert(addr, size);
    }
    return relaxed;
  }

  // memory_elf.cpp
  template <int W> address_type<W> Memory<W>::elf_base_address(address_t offset) const {
    if (this->m_is_dynamic) {
      const address_t vaddr_base = DYLINK_BASE;
      if (UNLIKELY(vaddr_base + offset < vaddr_base))
        throw MachineException(INVALID_PROGRAM, "Bogus virtual address + offset");
      return vaddr_base + offset;
    } else {
      return offset;
    }
  }

  template <int W>
  const typename Elf<W>::Sym *Memory<W>::elf_sym_index(const typename Elf::SectionHeader *shdr, uint32_t symidx) const {
    if (symidx >= shdr->sh_size / sizeof(typename Elf::Sym))
#ifdef __EXCEPTIONS
      throw MachineException(INVALID_PROGRAM, "ELF Symtab section index overflow");
#else
      std::abort();
#endif
    auto *symtab = this->elf_offset<typename Elf::Sym>(shdr->sh_offset);
    return &symtab[symidx];
  }

  template <int W> const typename Elf<W>::SectionHeader *Memory<W>::section_by_name(const std::string &name) const {
    auto &elf = *elf_header();
    const auto sh_end_offset = elf.e_shoff + elf.e_shnum * sizeof(typename Elf::SectionHeader);

    if (elf.e_shoff > m_binary.size())
      throw MachineException(INVALID_PROGRAM, "Invalid section header offset", elf.e_shoff);
    if (sh_end_offset < elf.e_shoff || sh_end_offset > m_binary.size())
      throw MachineException(INVALID_PROGRAM, "Invalid section header offset", sh_end_offset);
    if (elf.e_shnum == 0 || elf.e_shnum > 64)
      throw MachineException(INVALID_PROGRAM, "Invalid section header count", elf.e_shnum);
    const auto *shdr = elf_offset<typename Elf::SectionHeader>(elf.e_shoff);

    if (elf.e_shstrndx >= elf.e_shnum) throw MachineException(INVALID_PROGRAM, "Invalid section header strtab index");

    const auto &shstrtab = shdr[elf.e_shstrndx];
    const char *strings = elf_offset<char>(shstrtab.sh_offset);
    const char *endptr = m_binary.data() + m_binary.size();

    for (auto i = 0; i < elf.e_shnum; i++) {
      // Bounds-check and overflow-check on sh_name from strtab sh_offset
      const auto name_offset = shstrtab.sh_offset + shdr[i].sh_name;
      if (name_offset < shstrtab.sh_offset || name_offset >= m_binary.size())
        throw MachineException(INVALID_PROGRAM, "Invalid ELF string offset");

      const char *shname = &strings[shdr[i].sh_name];
      const size_t len = strnlen(shname, endptr - shname);
      if (len != name.size()) continue;

      if (strncmp(shname, name.c_str(), len) == 0) {
        return &shdr[i];
      }
    }
    return nullptr;
  }

  template <int W> const typename Elf<W>::Sym *Memory<W>::resolve_symbol(std::string_view name) const {
    if (UNLIKELY(m_binary.empty())) return nullptr;
    const auto *sym_hdr = section_by_name(".symtab");
    if (UNLIKELY(sym_hdr == nullptr)) return nullptr;
    const auto *str_hdr = section_by_name(".strtab");
    if (UNLIKELY(str_hdr == nullptr)) return nullptr;
    // ELF with no symbols
    if (UNLIKELY(sym_hdr->sh_size == 0)) return nullptr;

    const auto *symtab = elf_sym_index(sym_hdr, 0);
    const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf::Sym);
    const char *strtab = elf_offset<char>(str_hdr->sh_offset);

    for (size_t i = 0; i < symtab_ents; i++) {
      const char *symname = &strtab[symtab[i].st_name];
      if (name.compare(symname) == 0) {
        return &symtab[i];
      }
    }
    return nullptr;
  }

  template <int W> std::vector<const char *> Memory<W>::all_symbols() const {
    std::vector<const char *> symbols;
    if (UNLIKELY(m_binary.empty())) return symbols;
    const auto *sym_hdr = section_by_name(".symtab");
    const auto *str_hdr = section_by_name(".strtab");
    if (UNLIKELY(sym_hdr == nullptr || str_hdr == nullptr)) return symbols;
    // ELF with no symbols
    if (UNLIKELY(sym_hdr->sh_size == 0)) return symbols;

    const auto *symtab = elf_sym_index(sym_hdr, 0);
    const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf::Sym);
    const char *strtab = elf_offset<char>(str_hdr->sh_offset);
    symbols.reserve(symtab_ents);

    for (size_t i = 0; i < symtab_ents; i++) {
      const char *symname = &strtab[symtab[i].st_name];
      symbols.push_back(symname);
    }
    return symbols;
  }

  template <int W>
  std::vector<std::string_view> Memory<W>::all_unmangled_function_symbols(const std::string &prefix) const {
    std::vector<std::string_view> symbols;
    if (UNLIKELY(m_binary.empty())) return symbols;
    const auto *sym_hdr = section_by_name(".symtab");
    const auto *str_hdr = section_by_name(".strtab");
    if (UNLIKELY(sym_hdr == nullptr || str_hdr == nullptr)) return symbols;
    // ELF with no symbols
    if (UNLIKELY(sym_hdr->sh_size == 0)) return symbols;

    const auto *symtab = elf_sym_index(sym_hdr, 0);
    const size_t symtab_ents = sym_hdr->sh_size / sizeof(typename Elf::Sym);
    const char *strtab = elf_offset<char>(str_hdr->sh_offset);
    symbols.reserve(symtab_ents);

    for (size_t i = 0; i < symtab_ents; i++) {
      const char *symname = &strtab[symtab[i].st_name];
      if (Elf::SymbolType(symtab[i].st_info) == Elf::STT_FUNC && Elf::SymbolBind(symtab[i].st_info) != Elf::STB_WEAK) {
        std::string_view symview(symname);
        // Detect if the symbol is unmangled (no _Z prefix)
        if (symview.size() > 2 && !(symview[0] == '_' && symview[1] == 'Z')) {
          if (prefix.empty() || symview.compare(0, prefix.size(), prefix) == 0) symbols.push_back(symview);
        }
      }
    }
    return symbols;
  }

  template <int W> std::vector<std::string_view> Memory<W>::elf_comments() const {
    std::vector<std::string_view> comments;
    if (UNLIKELY(m_binary.empty())) return comments;
    const auto *hdr = elf_header();
    if (UNLIKELY(hdr == nullptr)) return comments;
    const auto *shdr = section_by_name(".comment");
    if (UNLIKELY(shdr == nullptr)) return comments;
    // ELF with no comments
    if (UNLIKELY(shdr->sh_size == 0)) return comments;

    const char *strtab = elf_offset<char>(shdr->sh_offset);
    const char *end = strtab + shdr->sh_size;
    const char *binary_end = m_binary.data() + m_binary.size(); // MSVC doesn't like m_binary.end()

    if (end < strtab || end > binary_end) throw MachineException(INVALID_PROGRAM, "Invalid ELF comment section");
    // Check if the comment section is null-terminated at the end
    if (UNLIKELY(end[-1] != '\0')) throw MachineException(INVALID_PROGRAM, "Invalid ELF comment section");
    // Use string_view to find each null-terminated comment
    while (strtab < end) {
      std::string_view comment(strtab);
      if (comment.empty()) {
        strtab++;
        continue;
      }
      comments.push_back(comment);
      strtab += comment.size() + 1;
      if (strtab >= binary_end) break;
    }
    return comments;
  }

  template <int W> static void elf_print_sym(const typename Elf<W>::Sym *sym) {
    if constexpr (W == 4) {
      printf("-> Sym is at 0x%" PRIX32 " with size %" PRIu32 ", type %u name %u\n", sym->st_value, sym->st_size,
             Elf<W>::SymbolType(sym->st_info), sym->st_name);
    } else {
      printf("-> Sym is at 0x%" PRIX64 " with size %" PRIu64 ", type %u name %u\n", (uint64_t)sym->st_value,
             sym->st_size, Elf<W>::SymbolType(sym->st_info), sym->st_name);
    }
  }

  template <int W> RISCV_INTERNAL void Memory<W>::relocate_section(const char *section_name, const char *sym_section) {
    using ElfRela = typename Elf::Rela;

    const auto *rela = section_by_name(section_name);
    if (rela == nullptr) return;
    const auto *dyn_hdr = section_by_name(sym_section);
    if (dyn_hdr == nullptr) return;
    const size_t rela_ents = rela->sh_size / sizeof(ElfRela);

    const auto rela_ents_offset = rela->sh_offset + rela_ents * sizeof(ElfRela);
    if (rela_ents_offset < rela->sh_offset || m_binary.size() < rela_ents_offset)
      throw MachineException(INVALID_PROGRAM, "Invalid ELF relocations");

    auto *rela_addr = elf_offset<ElfRela>(rela->sh_offset);
    for (size_t i = 0; i < rela_ents; i++) {
      size_t symidx;
      if constexpr (W == 4) symidx = Elf::RelaSym(rela_addr[i].r_info);
      else symidx = Elf::RelaSym(rela_addr[i].r_info);
      auto *sym = elf_sym_index(dyn_hdr, symidx);

      const uint8_t type = Elf::SymbolType(sym->st_info);
      if (true || type == Elf::STT_FUNC || type == Elf::STT_OBJECT) {
        if constexpr (false) {
          printf("Relocating rela %zu with sym idx %ld where 0x%lX -> 0x%lX\n", i, (long)symidx,
                 (long)rela_addr[i].r_offset, (long)sym->st_value);
          elf_print_sym<W>(sym);
        }
        const auto rtype = Elf::RelaType(rela_addr[i].r_info);
        static constexpr int R_RISCV_64 = 0x2;
        static constexpr int R_RISCV_RELATIVE = 0x3;
        static constexpr int R_RISCV_JUMPSLOT = 0x5;
        if (rtype == 0) {
          // Do nothing
        } else if (rtype == R_RISCV_64) {
          this->write<address_t>(elf_base_address(rela_addr[i].r_offset), elf_base_address(sym->st_value));
        } else if (rtype == R_RISCV_RELATIVE) {
          this->write<address_t>(elf_base_address(rela_addr[i].r_offset), sym->st_value);
        } else if (rtype == R_RISCV_JUMPSLOT) {
          // typedef struct {
          //	address_t r_offset;
          //	address_t r_info;
          // } Elf64_Rel;
          // printf("Relocating jumpslot %zu with sym idx %ld where 0x%lX -> 0x%lX\n",
          //		i, (long)symidx, (long)rela_addr[i].r_offset, (long)sym->st_value);
          // const auto* plt = section_by_name(".plt");
          // if (plt == nullptr)
          //	throw MachineException(INVALID_PROGRAM, "Missing .plt section for jumpslot relocation");
          // const auto* plt_addr = elf_offset<Elf64_Rel>(plt->sh_offset);
          // const Elf64_Rel& plt_entry = plt_addr[sym->st_value / sizeof(Elf64_Rel)];
          // const auto plt_address = elf_base_address(plt_entry.r_offset);
          //  Write the PLT address to the jumpslot
          // this->write<address_t>(elf_base_address(rela_addr[i].r_offset), plt_address);
        } else {
          throw MachineException(INVALID_PROGRAM, "Unknown relocation type", rtype);
        }
      }
    }
  }

  template <int W> RISCV_INTERNAL void Memory<W>::dynamic_linking(const typename Elf::Header &hdr) {
    (void)hdr;
    this->relocate_section(".rela.dyn", ".dynsym");
    this->relocate_section(".rela.plt", ".symtab");
  }
}
