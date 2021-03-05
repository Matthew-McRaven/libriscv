#include <include/syscall_helpers.hpp>
#include <include/native_heap.hpp>
using namespace riscv;
using namespace sas_alloc;
//#define HPRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define HPRINT(fmt, ...) /* */
//#define MPRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define MPRINT(fmt, ...) /* */

#ifndef NATIVE_SYSCALLS_BASE
#define NATIVE_SYSCALLS_BASE    1  /* They start at 1 */
#endif

template <int W>
static void setup_native_heap_syscalls(Machine<W>& machine,
	sas_alloc::Arena* arena)
{
	// Malloc n+0
	machine.install_syscall_handler(NATIVE_SYSCALLS_BASE+0,
	[arena] (auto& machine)
	{
		const size_t len = machine.template sysarg<address_type<W>>(0);
		auto data = arena->malloc(len);
		HPRINT("SYSCALL malloc(%zu) = 0x%X\n", len, data);
		machine.set_result(data);
	});
	// Calloc n+1
	machine.install_syscall_handler(NATIVE_SYSCALLS_BASE+1,
	[arena] (auto& machine)
	{
		const auto [count, size] =
			machine.template sysargs<address_type<W>, address_type<W>> ();
		const size_t len = count * size;
		auto data = arena->malloc(len);
		HPRINT("SYSCALL calloc(%u, %u) = 0x%X\n", count, size, data);
		if (data != 0) {
			// TODO: optimize this (CoW), **can throw**
			machine.memory.memset(data, 0, len);
		}
		machine.set_result(data);
	});
	// Realloc n+2
	machine.install_syscall_handler(NATIVE_SYSCALLS_BASE+2,
	[arena] (auto& machine)
	{
		const address_type<W> src = machine.template sysarg<address_type<W>>(0);
		const size_t newlen = machine.template sysarg<address_type<W>>(1);
		if (src != 0)
		{
			const size_t srclen = arena->size(src, false);
			if (srclen > 0)
			{
				if (srclen >= newlen) {
					return;
				}
				// XXX: really have to know what we are doing here
				// we are freeing in the hopes of getting the same chunk
				// back, in which case the copy could be completely skipped.
				arena->free(src);
				auto data = arena->malloc(newlen);
				HPRINT("SYSCALL realloc(0x%X:%zu, %zu) = 0x%X\n", src, srclen, newlen, data);
				// If the reallocation fails, return NULL
				if (data == 0) {
					arena->malloc(srclen);
					machine.set_result(data);
					return;
				}
				else if (data != src)
				{
					machine.memory.memcpy(data, machine, src, srclen);
				}
				machine.set_result(data);
				return;
			} else {
				HPRINT("SYSCALL realloc(0x%X:??, %zu) = 0x0\n", src, newlen);
			}
		} else {
			auto data = arena->malloc(newlen);
			HPRINT("SYSCALL realloc(0x0, %zu) = 0x%lX\n", newlen, (long) data);
			machine.set_result(data);
			return;
		}
		machine.set_result(0);
	});
	// Free n+3
	machine.install_syscall_handler(NATIVE_SYSCALLS_BASE+3,
	[arena] (auto& machine)
	{
		const auto ptr = machine.template sysarg<address_type<W>>(0);
		if (ptr != 0)
		{
			int ret = arena->free(ptr);
			HPRINT("SYSCALL free(0x%X) = %d\n", ptr, ret);
			machine.set_result(ret);
			if (ptr != 0x0 && ret < 0) {
				throw std::runtime_error("Possible double-free for freed pointer");
			}
			return;
		}
		HPRINT("SYSCALL free(0x0) = 0\n");
		machine.set_result(0);
		return;
	});
	// Meminfo n+4
	machine.install_syscall_handler(NATIVE_SYSCALLS_BASE+4,
	[arena] (auto& machine)
	{
		const auto dst = machine.template sysarg<address_type<W>>(0);
		struct Result {
			const uint32_t bf;
			const uint32_t bu;
			const uint32_t cu;
		} result = {
			.bf = (uint32_t) arena->bytes_free(),
			.bu = (uint32_t) arena->bytes_used(),
			.cu = (uint32_t) arena->chunks_used()
		};
		int ret = (dst != 0) ? 0 : -1;
		HPRINT("SYSCALL meminfo(0x%X) = %d\n", dst, ret);
		if (ret == 0) {
			machine.copy_to_guest(dst, &result, sizeof(result));
		}
		machine.set_result(ret);
	});
}

template <int W>
Arena* setup_native_heap_syscalls(Machine<W>& machine, uint64_t base, size_t max_memory)
{
	auto* arena = new sas_alloc::Arena(base, base + max_memory);
	machine.add_destructor_callback([arena] { delete arena; });

	setup_native_heap_syscalls<W> (machine, arena);
	return arena;
}
template <int W>
Arena* setup_native_heap_syscalls(Machine<W>& machine, uint64_t base, size_t max_memory,
	Function<void* (size_t)> constructor)
{
	sas_alloc::Arena* arena =
		(sas_alloc::Arena*) constructor(sizeof(sas_alloc::Arena));
	new (arena) sas_alloc::Arena(base, base + max_memory);

	setup_native_heap_syscalls<W> (machine, arena);
	return arena;
}

template <int W>
void setup_native_memory_syscalls(Machine<W>& machine, bool /*trusted*/)
{
	machine.install_syscall_handlers({
		{NATIVE_SYSCALLS_BASE+5, [] (auto& m) {
		// Memcpy n+5
		auto [dst, src, len] =
			m.template sysargs<address_type<W>, address_type<W>, address_type<W>> ();
		MPRINT("SYSCALL memcpy(%#X, %#X, %u)\n", dst, src, len);
		m.memory.memcpy(dst, m, src, len);
		m.increment_counter(2 * len);
		m.set_result(dst);
	}}, {NATIVE_SYSCALLS_BASE+6, [] (auto& m) {
		// Memset n+6
		const auto [dst, value, len] =
			m.template sysargs<address_type<W>, address_type<W>, address_type<W>> ();
		MPRINT("SYSCALL memset(%#X, %#X, %u)\n", dst, value, len);
		m.memory.memset(dst, value, len);
		m.increment_counter(len);
		m.set_result(dst);
	}}, {NATIVE_SYSCALLS_BASE+7, [] (auto& m) {
		// Memmove n+7
		auto [dst, src, len] =
			m.template sysargs<address_type<W>, address_type<W>, address_type<W>> ();
		MPRINT("SYSCALL memmove(%#lX, %#lX, %lu)\n",
			(long) dst, (long) src, (long) len);
		if (src < dst)
		{
			for (unsigned i = 0; i != len; i++) {
				m.memory.template write<uint8_t> (dst + i,
					m.memory.template read<uint8_t> (src + i));
			}
		} else {
			while (len-- != 0) {
				m.memory.template write<uint8_t> (dst + len,
					m.memory.template read<uint8_t> (src + len));
			}
		}
		m.increment_counter(2 * len);
		m.set_result(dst);
	}}, {NATIVE_SYSCALLS_BASE+8, [] (auto& m) {
		// Memcmp n+8
		auto [p1, p2, len] =
			m.template sysargs<address_type<W>, address_type<W>, address_type<W>> ();
		MPRINT("SYSCALL memcmp(%#X, %#X, %u)\n", p1, p2, len);
		m.increment_counter(2 * len);
		m.set_result(m.memory.memcmp(p1, p2, len));
	}}, {NATIVE_SYSCALLS_BASE+10, [] (auto& m) {
		// Strlen n+10
		auto [addr] = m.template sysargs<address_type<W>> ();
		uint32_t len = m.memory.strlen(addr);
		m.increment_counter(2 * len);
		m.set_result(len);
		MPRINT("SYSCALL strlen(%#lX) = %lu\n",
			(long) addr, (long) len);
	}}, {NATIVE_SYSCALLS_BASE+11, [] (auto& m) {
		// Strncmp n+11
		auto [a1, a2, maxlen] =
			m.template sysargs<address_type<W>, address_type<W>, uint32_t> ();
		MPRINT("SYSCALL strncmp(%#lX, %#lX, %u)\n", (long)a1, (long)a2, maxlen);
		uint32_t len = 0;
		while (len < maxlen) {
			const uint8_t v1 = m.memory.template read<uint8_t> (a1 ++);
			const uint8_t v2 = m.memory.template read<uint8_t> (a2 ++);
			if (v1 != v2 || v1 == 0) {
				m.increment_counter(2 + 2 * len);
				m.set_result(v1 - v2);
				return;
			}
			len ++;
		}
		m.increment_counter(2 + 2 * len);
		m.set_result(0);
	}}, {NATIVE_SYSCALLS_BASE+19, [] (auto& m) {
		// Print backtrace n+19
		m.memory.print_backtrace(
			[] (const char* buffer, size_t len) {
				printf("%.*s\n", (int)len, buffer);
			});
		m.set_result(0);
	}}});
}

uint64_t arena_malloc(sas_alloc::Arena* arena, const size_t len)
{
	return arena->malloc(len);
}
int arena_free(sas_alloc::Arena* arena, const uint64_t addr)
{
	return arena->free(addr);
}
void arena_transfer(const sas_alloc::Arena* from, sas_alloc::Arena* to)
{
	from->transfer(*to);
}

/* le sigh */
template Arena* setup_native_heap_syscalls<4>(Machine<4>&, uint64_t, size_t);
template Arena* setup_native_heap_syscalls<4>(Machine<4>& machine, uint64_t, size_t, Function<void* (size_t)>);
template void setup_native_memory_syscalls<4>(Machine<4>&, bool);

template Arena* setup_native_heap_syscalls<8>(Machine<8>&, uint64_t, size_t);
template Arena* setup_native_heap_syscalls<8>(Machine<8>& machine, uint64_t, size_t, Function<void* (size_t)>);
template void setup_native_memory_syscalls<8>(Machine<8>&, bool);
