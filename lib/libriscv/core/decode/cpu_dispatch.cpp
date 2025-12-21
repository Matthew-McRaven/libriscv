#include "../../machine.hpp"
#include "../instruction_counter.hpp"
#include "../instructions/rv32i_instr.hpp"
#include "../instructions/rvfd.hpp"
#include "decoder_cache_impl.hpp"
#include "threaded_bytecodes.hpp"
#ifdef RISCV_EXT_COMPRESSED
#include "../instructions/rvc.hpp"
#endif
#ifdef RISCV_EXT_VECTOR
#include "../instructions/rvv.hpp"
#endif

/**
 * This file is included by threaded_dispatch.cpp and bytecode_dispatch.cpp
 * It implements the logic for switch-based and threaded dispatch.
 * 
 * All dispatch modes share bytecode_impl.cpp
**/

namespace riscv
{
#define VIEW_INSTR() \
	auto instr = *(rv32i_instruction *)&decoder->instr;
#define VIEW_INSTR_AS(name, x) \
	auto &&name = *(x *)&decoder->instr;
#define NEXT_INSTR()                  \
	if constexpr (compressed_enabled) \
		decoder += 2;                 \
	else                              \
		decoder += 1;                 \
	EXECUTE_INSTR();
#define NEXT_C_INSTR() \
	decoder += 1;      \
	EXECUTE_INSTR();

#define NEXT_BLOCK(len, OF)                 \
	pc += len;                              \
	decoder += len >> DecoderCache<address_t>::SHIFT;              \
	if constexpr (FUZZING) /* Give OOB-aid to ASAN */      \
	decoder = &exec_decoder[pc >> DecoderCache<address_t>::SHIFT]; \
	if constexpr (OF) {						\
		if (UNLIKELY(counter.overflowed())) \
			goto check_jump;				\
	}										\
	pc += decoder->block_bytes();                            \
	counter.increment_counter(decoder->instruction_count()); \
	EXECUTE_INSTR();

#define SAFE_INSTR_NEXT(len)                  \
	pc += len;                                \
	decoder += len >> DecoderCache<address_t>::SHIFT; \
	counter.increment_counter(1);

#define NEXT_SEGMENT()                                       \
	decoder = &exec_decoder[pc >> DecoderCache<address_t>::SHIFT];  \
	pc += decoder->block_bytes();                            \
	counter.increment_counter(decoder->instruction_count()); \
	EXECUTE_INSTR();

#define PERFORM_BRANCH()                 \
	if constexpr (VERBOSE_JUMPS) fprintf(stderr, "Branch 0x%lX >= 0x%lX (decoder=%p)\n", long(pc), long(pc + fi.signed_imm()), decoder); \
	if (LIKELY(!counter.overflowed())) { \
		NEXT_BLOCK(fi.signed_imm(), false);     \
	}                                    \
	pc += fi.signed_imm();               \
	goto check_jump;

#define PERFORM_FORWARD_BRANCH()         \
	if constexpr (VERBOSE_JUMPS) fprintf(stderr, "Fw.Branch 0x%lX >= 0x%lX\n", long(pc), long(pc + fi.signed_imm())); \
	NEXT_BLOCK(fi.signed_imm(), false);

#define OVERFLOW_CHECKED_JUMP() \
	goto check_jump

template <AddressType address_t>
DISPATCH_ATTR bool CPU<address_t>::simulate(address_t pc, uint64_t inscounter, uint64_t maxcounter) {
  static constexpr auto W = sizeof(address_t);
  static constexpr uint32_t XLEN = W * 8;
	using addr_t  = address_t;
  using saddr_t = ToSignedAddress<addr_t>;

#ifdef DISPATCH_MODE_THREADED
#include "threaded_bytecode_array.hpp"
#endif

	DecodedExecuteSegment<address_t>* exec = this->m_exec;
	address_t current_begin = exec->exec_begin();
	address_t current_end   = exec->exec_end();

	DecoderData<address_t>* exec_decoder = exec->decoder_cache();
	DecoderData<address_t>* decoder;

	InstrCounter counter{inscounter, maxcounter};

	// We need an execute segment matching current PC
	if (UNLIKELY(!(pc >= current_begin && pc < current_end)))
		goto new_execute_segment;

continue_segment:
	decoder = &exec_decoder[pc >> DecoderCache<address_t>::SHIFT];

	pc += decoder->block_bytes();
	counter.increment_counter(decoder->instruction_count());

#ifdef DISPATCH_MODE_SWITCH_BASED

while (true) {
	switch (decoder->get_bytecode()) {
	#define INSTRUCTION(bc, lbl) case bc:

#else
	goto *computed_opcode[decoder->get_bytecode()];
	#define INSTRUCTION(bc, lbl) lbl:

#endif

#define DECODER()   (*decoder)
#define CPU()       (*this)
#define REG(x)      registers().get()[x]
#define REGISTERS() registers()
#define VECTORS()   registers().rvv()
#define MACHINE()   machine()

	/** Instruction handlers **/

#include "bytecode_impl.cpp"

INSTRUCTION(RV32I_BC_SYSTEM, rv32i_system) {
	VIEW_INSTR();
	// Make the current PC visible
	REGISTERS().pc = pc;
	// Make the instruction counters visible
	counter.apply(MACHINE());
	// Invoke SYSTEM
	MACHINE().system(instr);
	// Restore counters
	counter.retrieve_counters(MACHINE());
	if (UNLIKELY(counter.overflowed() || pc != REGISTERS().pc))
	{
		pc = REGISTERS().pc;
		goto check_jump;
	}
	// Overflow-check, next block
	NEXT_BLOCK(4, true);
}

INSTRUCTION(RV32I_BC_SYSCALL, rv32i_syscall) {
	// Make the current PC visible
	REGISTERS().pc = pc;
	// Make the instruction counter(s) visible
	counter.apply(MACHINE());
	// Invoke system call
	MACHINE().system_call(REG(REG_ECALL));
	// Restore counters
	counter.retrieve_counters(MACHINE());
	if (UNLIKELY(counter.overflowed() || pc != REGISTERS().pc))
	{
		// System calls are always full-length instructions
		if constexpr (VERBOSE_JUMPS) {
			if (pc != REGISTERS().pc)
			fprintf(stderr, "SYSCALL jump from 0x%lX to 0x%lX\n",
				long(pc), long(REGISTERS().pc + 4));
		}
		pc = REGISTERS().pc + 4;
		goto check_jump;
	}
	NEXT_BLOCK(4, false);
}

INSTRUCTION(RV32I_BC_STOP, rv32i_stop) {
	REGISTERS().pc = pc + 4;
	MACHINE().set_instruction_counter(counter.value());
	return true;
}

#ifdef DISPATCH_MODE_SWITCH_BASED
	default:
		goto execute_invalid;
	} // switch case
} // while loop

#endif

check_jump:
	if (UNLIKELY(counter.overflowed()))
		goto counter_overflow;

	if (LIKELY(pc - current_begin < current_end - current_begin))
		goto continue_segment;
	else
		goto new_execute_segment;

counter_overflow:
	registers().pc = pc;
	MACHINE().set_instruction_counter(counter.value());

	// Machine stopped normally?
	return counter.max() == 0;

	// Change to a new execute segment
new_execute_segment: {
		auto new_values = this->next_execute_segment(pc);
		exec = new_values.exec;
		pc   = new_values.pc;
		current_begin = exec->exec_begin();
		current_end   = exec->exec_end();
		exec_decoder  = exec->decoder_cache();
	}
	goto continue_segment;

execute_invalid:
	// Calculate the current PC from the decoder pointer
	pc = (decoder - exec_decoder) << DecoderCache<address_t>::SHIFT;
	// Check if the instruction is still invalid
	try {
		if (decoder->instr == 0 && MACHINE().memory.template read<uint16_t>(pc) != 0) {
			exec->set_stale(true);
			goto new_execute_segment;
		}
	} catch (...) {}
	MACHINE().set_instruction_counter(counter.value());
	registers().pc = pc;
	trigger_exception(ILLEGAL_OPCODE, decoder->instr);

#ifdef RISCV_LIBTCC
handle_rethrow_exception:
	// We have an exception, so we need to rethrow it
	const auto except = CPU().current_exception();
	CPU().clear_current_exception();
	std::rethrow_exception(except);
#endif

} // CPU::simulate_XXX()
INSTANTIATE_32_IF_ENABLED(CPU);
INSTANTIATE_64_IF_ENABLED(CPU);
} // riscv
