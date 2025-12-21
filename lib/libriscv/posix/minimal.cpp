#include <libriscv/machine.hpp>
#include <cstdio>
#define SYSPRINT(fmt, ...) /* fmt */

namespace riscv
{
	template <AddressType address_t>
	static void syscall_stub_zero(Machine<address_t>& machine)
	{
		SYSPRINT("SYSCALL stubbed (zero): %d\n", (int)machine.cpu.reg(17));
		machine.set_result(0);
	}

	template <AddressType address_t>
	static void syscall_stub_nosys(Machine<address_t>& machine)
	{
		SYSPRINT("SYSCALL stubbed (nosys): %d\n", (int)machine.cpu.reg(17));
		machine.set_result(-38); // ENOSYS
	}

	template <AddressType address_t>
	static void syscall_ebreak(riscv::Machine<address_t>& machine)
	{
		printf("\n>>> EBREAK at %#lX\n", (long)machine.cpu.pc());
		throw MachineException(UNHANDLED_SYSCALL, "EBREAK instruction");
	}

	template<AddressType address_t>
	static void syscall_write(Machine<address_t>& machine)
	{
		const int vfd = machine.template sysarg<int>(0);
		const auto address = machine.sysarg(1);
		const size_t len = machine.sysarg(2);
		SYSPRINT("SYSCALL write, fd: %d addr: 0x%lX, len: %zu\n",
				vfd, (long) address, len);
		// We only accept standard output pipes, for now :)
		if (vfd == 1 || vfd == 2) {
			// Zero-copy retrieval of buffers (64kb)
			riscv::vBuffer buffers[16];
			const size_t cnt =
				machine.memory.gather_buffers_from_range(16, buffers, address, len);
			for (size_t i = 0; i < cnt; i++) {
				machine.print(buffers[i].ptr, buffers[i].len);
			}
			machine.set_result(len);
			return;
		}
		machine.set_result(-EBADF);
	}

	template <AddressType address_t>
	static void syscall_exit(Machine<address_t>& machine)
	{
		// Stop sets the max instruction counter to zero, allowing most
		// instruction loops to end. It is, however, not the only way
		// to exit a program. Tighter integrations with the library should
		// provide their own methods.
		machine.stop();
	}

	template <AddressType address_t>
	static void syscall_brk(Machine<address_t>& machine)
	{
		auto new_end = machine.sysarg(0);
		if (new_end > machine.memory.heap_address() + Memory<address_t>::BRK_MAX) {
			new_end = machine.memory.heap_address() + Memory<address_t>::BRK_MAX;
		} else if (new_end < machine.memory.heap_address()) {
			new_end = machine.memory.heap_address();
		}

		SYSPRINT("SYSCALL brk, new_end: 0x%lX\n", (long)new_end);
		machine.set_result(new_end);
	}

	template <AddressType address_t>
	void Machine<address_t>::setup_minimal_syscalls()
	{
		install_syscall_handler(SYSCALL_EBREAK, syscall_ebreak<address_t>);
		install_syscall_handler(57, syscall_stub_zero<address_t>);  // close
		install_syscall_handler(62, syscall_stub_nosys<address_t>); // lseek
		install_syscall_handler(64, syscall_write<address_t>);
		install_syscall_handler(80, syscall_stub_nosys<address_t>); // fstat
		install_syscall_handler(93, syscall_exit<address_t>);
		install_syscall_handler(214, syscall_brk<address_t>);
	}

#ifdef RISCV_32I
  template void Machine<uint32_t>::setup_minimal_syscalls();
#endif
#ifdef RISCV_64I
  template void Machine<uint64_t>::setup_minimal_syscalls();
#endif
} // riscv
