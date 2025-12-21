#include "common.hpp"
#include "cpu.hpp"
#include "riscvbase.hpp"
#include "threaded_bytecodes.hpp"

namespace riscv {
// Use a trick to access the Machine directly on g++/clang, Linux-only for now
#if (defined(__GNUG__) || defined(__clang__)) && defined(__linux__)
template <int W> RISCV_ALWAYS_INLINE inline
Machine<W>& CPU<W>::machine() noexcept { return *reinterpret_cast<Machine<W>*> (this); }
template <int W> RISCV_ALWAYS_INLINE inline
const Machine<W>& CPU<W>::machine() const noexcept { return *reinterpret_cast<const Machine<W>*> (this); }
#else
template <int W> RISCV_ALWAYS_INLINE inline
Machine<W>& CPU<W>::machine() noexcept { return this->m_machine; }
template <int W> RISCV_ALWAYS_INLINE inline
const Machine<W>& CPU<W>::machine() const noexcept { return this->m_machine; }
#endif

template <int W> RISCV_ALWAYS_INLINE inline
Memory<W>& CPU<W>::memory() noexcept { return machine().memory; }
template <int W> RISCV_ALWAYS_INLINE inline
const Memory<W>& CPU<W>::memory() const noexcept { return machine().memory; }

template <int W>
inline CPU<W>::CPU(Machine<W>& machine)
	: m_machine { machine }, m_exec(empty_execute_segment().get())
{
}
template <int W>
inline void CPU<W>::reset_stack_pointer() noexcept
{
	// initial stack location
	this->reg(2) = machine().memory.stack_initial();
}

template<int W>
inline void CPU<W>::jump(const address_t dst)
{
	// it's possible to jump to a misaligned address
	if constexpr (!compressed_enabled) {
		if (UNLIKELY(dst & 0x3)) {
			trigger_exception(MISALIGNED_INSTRUCTION, dst);
		}
	} else {
		if (UNLIKELY(dst & 0x1)) {
			trigger_exception(MISALIGNED_INSTRUCTION, dst);
		}
	}
	this->registers().pc = dst;
}

template<int W>
inline void CPU<W>::aligned_jump(const address_t dst) noexcept
{
	this->registers().pc = dst;
}

template <int W> inline void CPU<W>::increment_pc(int delta) noexcept { registers().pc += delta; }

// cpu.cpp
template <int W>
CPU<W>::CPU(Machine<W> &machine, const Machine<W> &other) : m_machine{machine}, m_exec(other.cpu.m_exec) {
  // Copy all registers except vectors
  // Users can still copy vector registers by assigning to registers().rvv().
  this->registers().copy_from(Registers<W>::Options::NoVectors, other.cpu.registers());
}
template <int W> void CPU<W>::reset() {
  this->m_regs = {};
  this->reset_stack_pointer();
  // We can't jump if there's been no ELF loader
  if (!current_execute_segment().empty()) {
    const auto initial_pc = machine().memory.start_address();
    // Check if the initial PC is executable, unless
    // the execute segment is marked as execute-only.
    if (!current_execute_segment().is_execute_only()) {
      const auto &page = machine().memory.get_exec_pageno(initial_pc / riscv::Page::size());
      if (UNLIKELY(!page.attr.exec)) trigger_exception(EXECUTION_SPACE_PROTECTION_FAULT, initial_pc);
    }
    // This function will (at most) validate the execute segment
    this->jump(initial_pc);
  }
}

template <int W> RISCV_NOINLINE typename CPU<W>::NextExecuteReturn CPU<W>::next_execute_segment(address_t pc) {
  static constexpr int MAX_RESTARTS = 4;
  int restarts = 0;
restart_next_execute_segment:

  // Find previously decoded execute segment
  this->m_exec = machine().memory.exec_segment_for(pc).get();
  if (LIKELY(!this->m_exec->empty() && !this->m_exec->is_stale())) {
    return {this->m_exec, pc};
  }

  // We absolutely need to write PC here because even read-fault handlers
  // like get_pageno() slowpaths could be reading PC.
  this->registers().pc = pc;

  // Immediately look at the page in order to
  // verify execute and see if it has a trap handler
  address_t base_pageno = pc / Page::size();
  address_t end_pageno = base_pageno + 1;

  // Check for +exec
  const auto &current_page = machine().memory.get_pageno(base_pageno);
  if (UNLIKELY(!current_page.attr.exec)) {
    this->m_fault(*this, current_page);
    pc = this->pc();

    if (UNLIKELY(++restarts == MAX_RESTARTS)) trigger_exception(EXECUTION_LOOP_DETECTED, pc);

    goto restart_next_execute_segment;
  }

  // Check for trap
  if (UNLIKELY(current_page.has_trap())) {
    // We pass PC as offset
    current_page.trap(pc & (Page::size() - 1), TRAP_EXEC, pc);
    pc = this->pc();

    // If PC changed, we will restart the process
    if (pc / Page::size() != base_pageno) {
      if (UNLIKELY(++restarts == MAX_RESTARTS)) trigger_exception(EXECUTION_LOOP_DETECTED, pc);

      goto restart_next_execute_segment;
    }
  }

  // Evict stale execute segments
  if (this->m_exec->is_stale()) {
    machine().memory.evict_execute_segment(*this->m_exec);
  }

  // Find decoded execute segment via override
  // If it returns empty, we build a new execute segment
  auto &next = this->m_override_exec(*this);
  if (!next.empty()) {
    this->m_exec = &next;
    return {this->m_exec, this->registers().pc};
  }

  // Find the earliest execute page in new segment
  const uint8_t *base_page_data = current_page.data();

  while (base_pageno > 0) {
    const auto &page = machine().memory.get_pageno(base_pageno - 1);
    if (page.attr.exec) {
      base_pageno -= 1;
      base_page_data = page.data();
    } else break;
  }

  // Find the last execute page in segment
  const uint8_t *end_page_data = current_page.data();
  while (end_pageno != 0) {
    const auto &page = machine().memory.get_pageno(end_pageno);
    if (page.attr.exec) {
      end_pageno += 1;
      end_page_data = page.data();
    } else break;
  }

  if (UNLIKELY(end_pageno <= base_pageno)) throw MachineException(INVALID_PROGRAM, "Failed to create execute segment");
  const size_t n_pages = end_pageno - base_pageno;
  end_page_data += Page::size();
  const bool sequential = end_page_data == base_page_data + n_pages * Page::size();
  // Check if it's likely a JIT-compiled area
  const bool is_likely_jit = current_page.attr.exec && current_page.attr.write;

  // Allocate full execute area
  if (!sequential) {
    std::unique_ptr<uint8_t[]> area(new uint8_t[n_pages * Page::size()]);
    // Copy from each individual page
    for (address_t p = base_pageno; p < end_pageno; p++) {
      // Cannot use get_exec_pageno here as we may need
      // access to read fault handler.
      auto &page = machine().memory.get_pageno(p);
      const size_t offset = (p - base_pageno) * Page::size();
      std::memcpy(area.get() + offset, page.data(), Page::size());
    }

    // Decode and store it for later
    return {&this->init_execute_area(area.get(), base_pageno * Page::size(), n_pages * Page::size(), is_likely_jit),
            pc};
  } else {
    // We can use the sequential execute segment directly
    return {&this->init_execute_area(base_page_data, base_pageno * Page::size(), n_pages * Page::size(), is_likely_jit),
            pc};
  }
} // CPU::next_execute_segment

template <int W>
RISCV_NOINLINE RISCV_INTERNAL typename CPU<W>::format_t CPU<W>::read_next_instruction_slowpath() const {
  // Fallback: Read directly from page memory
  const auto pageno = this->pc() / address_t(Page::size());
  const auto &page = machine().memory.get_exec_pageno(pageno);
  if (UNLIKELY(!page.attr.exec)) {
    trigger_exception(EXECUTION_SPACE_PROTECTION_FAULT, this->pc());
  }
  const auto offset = this->pc() & (Page::size() - 1);
  format_t instruction;

  if (LIKELY(offset <= Page::size() - 4)) {
    instruction.whole = uint32_t(*(UnderAlign32 *)(page.data() + offset));
    return instruction;
  }
  // It's not possible to jump to a misaligned address,
  // so there is necessarily 16-bit left of the page now.
  instruction.whole = *(uint16_t *)(page.data() + offset);

  // If it's a 32-bit instruction at a page border, we need
  // to get the next page, and then read the upper half
  if (UNLIKELY(instruction.is_long())) {
    const auto &slow_page = machine().memory.get_exec_pageno(pageno + 1);
    instruction.half[1] = *(uint16_t *)slow_page.data();
  }

  return instruction;
}

template <int W> bool CPU<W>::is_executable(address_t addr) const noexcept { return m_exec->is_within(addr); }

template <int W> typename CPU<W>::format_t CPU<W>::read_next_instruction() const {
  if (LIKELY(this->is_executable(this->pc()))) {
    auto *exd = m_exec->exec_data(this->pc());
    return format_t{*(uint32_t *)exd};
  }

  return read_next_instruction_slowpath();
}

template <int W> static inline rv32i_instruction decode_safely(const uint8_t *exec_seg_data, address_type<W> pc) {
  // Instructions may be unaligned with C-extension
  // On amd64 we take the cost, because it's faster
#if defined(RISCV_EXT_COMPRESSED) && !defined(__x86_64__)
  return rv32i_instruction{*(UnderAlign32 *)&exec_seg_data[pc]};
#else  // aligned/unaligned loads
  return rv32i_instruction{*(uint32_t *)&exec_seg_data[pc]};
#endif // aligned/unaligned loads
}

template <int W> RISCV_HOT_PATH() void CPU<W>::simulate_precise() {
  // Decoded segments are always faster
  // So, always have at least the current segment
  if (!is_executable(this->pc())) {
    this->next_execute_segment(this->pc());
  }

  auto *exec = this->m_exec;
restart_precise_sim:
  auto *exec_seg_data = exec->exec_data();

  for (; machine().instruction_counter() < machine().max_instructions(); machine().increment_counter(1)) {

    auto pc = this->pc();

    // TODO: This can me made much faster
    if (UNLIKELY(!exec->is_within(pc))) {
      // This will produce a sequential execute segment for the unknown area
      // If it is not executable, it will throw an execute space protection fault
      auto new_values = this->next_execute_segment(pc);
      exec = new_values.exec;
      pc = new_values.pc;
      goto restart_precise_sim;
    }

    auto instruction = decode_safely<W>(exec_seg_data, pc);
    this->execute(instruction);

    // increment PC
    if constexpr (compressed_enabled) registers().pc += instruction.length();
    else registers().pc += 4;
  } // while not stopped

} // CPU::simulate_precise

template <int W> void CPU<W>::step_one(bool use_instruction_counter) {
  // Read, decode & execute instructions directly
  auto instruction = this->read_next_instruction();
  this->execute(instruction);

  if constexpr (compressed_enabled) registers().pc += instruction.length();
  else registers().pc += 4;

  machine().increment_counter(use_instruction_counter ? 1 : 0);
}

template <int W>
address_type<W> CPU<W>::preempt_internal(Registers<W> &old_regs, bool Throw, bool store_regs, address_t pc,
                                         uint64_t max_instr) {
  auto &m = machine();
  const auto prev_max = m.max_instructions();
  try {
    // execute by extending the max instruction counter (resuming)
    // WARNING: Do not change this, as resumption is required in
    // order for sandbox integrity. Repeatedly invoking preemption
    // should lead to timeouts on either preempt() *or* the caller.
    m.simulate_with(m.instruction_counter() + max_instr, m.instruction_counter(), pc);
  } catch (...) {
    m.set_max_instructions(prev_max);
    if (store_regs) {
      this->registers() = old_regs;
    }
    if (Throw) throw; // Only rethrow if we're supposed to forward exceptions
  }
  // restore registers and return value
  m.set_max_instructions(prev_max);
  const auto retval = this->reg(REG_ARG0);
  if (store_regs) {
    this->registers() = old_regs;
  }
  return retval;
}

template <int W> RISCV_COLD_PATH() void CPU<W>::trigger_exception(int intr, address_t data) {
  switch (intr) {
  case INVALID_PROGRAM: throw MachineException(intr, "Machine not initialized", data);
  case ILLEGAL_OPCODE: throw MachineException(intr, "Illegal opcode executed", data);
  case ILLEGAL_OPERATION: throw MachineException(intr, "Illegal operation during instruction decoding", data);
  case PROTECTION_FAULT: throw MachineException(intr, "Protection fault", data);
  case EXECUTION_SPACE_PROTECTION_FAULT: throw MachineException(intr, "Execution space protection fault", data);
  case EXECUTION_LOOP_DETECTED: throw MachineException(intr, "Execution loop detected", data);
  case MISALIGNED_INSTRUCTION:
    // NOTE: only check for this when jumping or branching
    throw MachineException(intr, "Misaligned instruction executed", data);
  case INVALID_ALIGNMENT: throw MachineException(intr, "Invalid alignment for address", data);
  case UNIMPLEMENTED_INSTRUCTION: throw MachineException(intr, "Unimplemented instruction executed", data);
  case DEADLOCK_REACHED: throw MachineException(intr, "Atomics deadlock reached", data);
  case OUT_OF_MEMORY: throw MachineException(intr, "Out of memory", data);

  default: throw MachineException(UNKNOWN_EXCEPTION, "Unknown exception", intr);
  }
}

template <int W> RISCV_COLD_PATH() std::string CPU<W>::to_string(format_t bits) const {
  return to_string(bits, decode(bits));
}

template <int W> RISCV_COLD_PATH() std::string CPU<W>::current_instruction_to_string() const {
  format_t instruction;
  try {
    instruction = this->read_next_instruction();
  } catch (...) {
    instruction = format_t{};
  }
  return to_string(instruction, decode(instruction));
}

template <int W> RISCV_COLD_PATH() std::string Registers<W>::flp_to_string() const {
  char buffer[800];
  int len = 0;
  for (int i = 0; i < 32; i++) {
    auto &src = this->getfl(i);
    const char T = (src.i32[1] == 0) ? 'S' : 'D';
    if constexpr (true) {
      double val = (src.i32[1] == 0) ? src.f32[0] : src.f64;
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%c%+.2f] ", RISCV::flpname(i), T, val);
    } else {
      if (src.i32[1] == 0) {
        double val = src.f64;
        len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%c0x%lX] ", RISCV::flpname(i), T, *(int64_t *)&val);
      } else {
        float val = src.f32[0];
        len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%c0x%X] ", RISCV::flpname(i), T, *(int32_t *)&val);
      }
    }
    if (i % 5 == 4) {
      len += snprintf(buffer + len, sizeof(buffer) - len, "\n");
    }
  }
  len += snprintf(buffer + len, sizeof(buffer) - len, "[FFLAGS\t0x%X] ", m_fcsr.fflags);
  return std::string(buffer, len);
}

// decode_bytecodes.cpp
template <int W> size_t CPU<W>::computed_index_for(rv32i_instruction instr) noexcept {
#ifdef RISCV_EXT_COMPRESSED
  if (instr.length() == 2) {
    // RISC-V Compressed Extension
    const rv32c_instruction ci{instr};
#define CI_CODE(x, y) ((x << 13) | (y))
    switch (ci.opcode()) {
    case CI_CODE(0b000, 0b00):
      // if all bits are zero, it's an illegal instruction
      if (ci.whole != 0x0) {
        return RV32C_BC_ADDI; // C.ADDI4SPN
      }
      return RV32I_BC_INVALID;
    case CI_CODE(0b001, 0b00):
    case CI_CODE(0b010, 0b00):
    case CI_CODE(0b011, 0b00): {
      if (ci.CL.funct3 == 0x1) {
        return RV32C_BC_FUNCTION; // C.FLD
      } else if (ci.CL.funct3 == 0x2) {
        return RV32C_BC_LDW; // C.LW
      } else if (ci.CL.funct3 == 0x3) {
        if constexpr (sizeof(address_t) == 8) {
          return RV32C_BC_FUNCTION; // C.LD
        } else {
          return RV32C_BC_FUNCTION; // C.FLW
        }
      }
      return RV32C_BC_FUNCTION; // C.UNIMP
    }
    // RESERVED: 0b100, 0b00
    case CI_CODE(0b101, 0b00):
    case CI_CODE(0b110, 0b00):
    case CI_CODE(0b111, 0b00): {
      switch (ci.CS.funct3) {
      case 4: return RV32C_BC_FUNCTION; // C.UNIMP
      case 5: return RV32C_BC_FUNCTION; // C.FSD
      case 6: return RV32C_BC_STW;      // C.SW
      case 7:                           // C.SD / C.FSW
        if constexpr (sizeof(address_t) == 8) {
          return RV32C_BC_STD; // C.SD
        } else {
          return RV32C_BC_FUNCTION; // C.FSW
        }
      }
      return RV32C_BC_FUNCTION; // C.UNIMP?
    }
    case CI_CODE(0b000, 0b01):
      if (ci.CI.rd != 0) {
        return RV32C_BC_ADDI; // C.ADDI
      }
      return RV32C_BC_FUNCTION; // C.NOP
    case CI_CODE(0b010, 0b01):
      if (ci.CI.rd != 0) {
        return RV32C_BC_LI; // C.LI
      }
      return RV32C_BC_FUNCTION; // C.NOP
    case CI_CODE(0b011, 0b01):
      if (ci.CI.rd == 2) {
        return RV32C_BC_ADDI; // C.ADDI16SP
      } else if (ci.CI.rd != 0) {
        return RV32C_BC_FUNCTION; // C.LUI
      }
      return RV32C_BC_FUNCTION; // ILLEGAL
    case CI_CODE(0b001, 0b01):
      if constexpr (W >= 8) {
        return RV32C_BC_JAL_ADDIW; // C.ADDIW
      } else {
        return RV32C_BC_JAL_ADDIW; // C.JAL
      }
    case CI_CODE(0b101, 0b01): // C.JMP
      return RV32C_BC_JMP;
    case CI_CODE(0b110, 0b01): // C.BEQZ
      return RV32C_BC_BEQZ;
    case CI_CODE(0b111, 0b01): // C.BNEZ
      return RV32C_BC_BNEZ;
    // Quadrant 2
    case CI_CODE(0b000, 0b10):
    case CI_CODE(0b001, 0b10):
    case CI_CODE(0b010, 0b10):
    case CI_CODE(0b011, 0b10): {
      if (ci.CI.funct3 == 0x0 && ci.CI.rd != 0) {
        return RV32C_BC_SLLI; // C.SLLI
      } else if (ci.CI2.funct3 == 0x1) {
        return RV32C_BC_FUNCTION; // C.FLDSP
      } else if (ci.CI2.funct3 == 0x2 && ci.CI2.rd != 0) {
        return RV32C_BC_LDW; // C.LWSP
      } else if (ci.CI2.funct3 == 0x3) {
        if constexpr (sizeof(address_t) == 8) {
          if (ci.CI2.rd != 0) {
            return RV32C_BC_LDD; // C.LDSP
          }
        } else {
          return RV32C_BC_FUNCTION; // C.FLWSP
        }
      } else if (ci.CI.rd == 0) {
        return RV32C_BC_FUNCTION; // C.HINT
      }
      return RV32C_BC_FUNCTION; // C.UNIMP?
    }
    case CI_CODE(0b100, 0b01): { // C1 ALU OPS
      switch (ci.CA.funct6 & 0x3) {
      case 0x0: return RV32C_BC_SRLI;     // C.SRLI
      case 0x1: return RV32C_BC_FUNCTION; // C.SRAI
      case 0x2: return RV32C_BC_ANDI;     // C.ANDI
      case 0x3:                           // More ALU ops
        switch (ci.CA.funct2 | (ci.CA.funct6 & 0x4)) {
        case 0x0: return RV32C_BC_FUNCTION; // C.SUB
        case 0x1: return RV32C_BC_XOR;      // C.XOR
        case 0x2: return RV32C_BC_OR;       // C.OR
        case 0x3: return RV32C_BC_FUNCTION; // C.AND
        default: return RV32C_BC_FUNCTION;
        }
      default: return RV32C_BC_FUNCTION;
      }
    }
    case CI_CODE(0b100, 0b10): {
      const bool topbit = ci.whole & (1 << 12);
      if (!topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0) {
        return RV32C_BC_JR; // C.JR rd
      } else if (topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0) {
        return RV32C_BC_JALR;                                  // C.JALR ra, rd+0
      } else if (!topbit && ci.CR.rd != 0 && ci.CR.rs2 != 0) { // MV rd, rs2
        return RV32C_BC_MV;                                    // C.MV
      } else if (ci.CR.rd != 0) {                              // ADD rd, rd + rs2
        return RV32C_BC_ADD;                                   // C.ADD
      } else if (topbit && ci.CR.rd == 0 && ci.CR.rs2 == 0) {  // EBREAK
        return RV32C_BC_FUNCTION;                              // C.EBREAK
      }
      return RV32C_BC_FUNCTION; // C.UNIMP?
    }
    case CI_CODE(0b101, 0b10):
    case CI_CODE(0b110, 0b10):
    case CI_CODE(0b111, 0b10): {
      if (ci.CSS.funct3 == 5) {
        return RV32C_BC_FUNCTION; // FSDSP
      } else if (ci.CSS.funct3 == 6) {
        return RV32C_BC_STW; // SWSP
      } else if (ci.CSS.funct3 == 7) {
        if constexpr (W == 8) {
          return RV32C_BC_STD; // SDSP
        } else {
          return RV32C_BC_FUNCTION; // FSWSP
        }
      }
      return RV32C_BC_FUNCTION; // C.UNIMP?
    }
    default: return RV32C_BC_FUNCTION;
    }
  }
#endif

  switch (instr.opcode()) {
  case RV32I_LOAD:
    // XXX: Support dummy loads
    if (instr.Itype.rd == 0) return RV32I_BC_FUNCTION;
    switch (instr.Itype.funct3) {
    case 0x0: // LD.B
      return RV32I_BC_LDB;
    case 0x1: // LD.H
      return RV32I_BC_LDH;
    case 0x2: // LD.W
      return RV32I_BC_LDW;
#ifdef RISCV_64I
    case 0x3:
      if constexpr (W >= 8) {
        return RV32I_BC_LDD;
      }
      return RV32I_BC_INVALID;
#endif
    case 0x4: // LD.BU
      return RV32I_BC_LDBU;
    case 0x5: // LD.HU
      return RV32I_BC_LDHU;
#ifdef RISCV_64I
    case 0x6: // LD.WU
      if constexpr (W >= 8) {
        return RV32I_BC_LDWU;
      }
      return RV32I_BC_INVALID;
#endif
    default: return RV32I_BC_INVALID;
    }
  case RV32I_STORE:
    switch (instr.Stype.funct3) {
    case 0x0: // SD.B
      return RV32I_BC_STB;
    case 0x1: // SD.H
      return RV32I_BC_STH;
    case 0x2: // SD.W
      return RV32I_BC_STW;
#ifdef RISCV_64I
    case 0x3:
      if constexpr (W >= 8) {
        return RV32I_BC_STD;
      }
      return RV32I_BC_INVALID;
#endif
    default: return RV32I_BC_INVALID;
    }
  case RV32I_BRANCH:
    switch (instr.Btype.funct3) {
    case 0x0: // BEQ
      return RV32I_BC_BEQ;
    case 0x1: // BNE
      return RV32I_BC_BNE;
    case 0x4: // BLT
      return RV32I_BC_BLT;
    case 0x5: // BGE
      return RV32I_BC_BGE;
    case 0x6: // BLTU
      return RV32I_BC_BLTU;
    case 0x7: // BGEU
      return RV32I_BC_BGEU;
    default: return RV32I_BC_INVALID;
    }
  case RV32I_LUI:
    if (instr.Utype.rd == 0) return RV32I_BC_FUNCTION;
    return RV32I_BC_LUI;
  case RV32I_AUIPC:
    if (instr.Utype.rd == 0) return RV32I_BC_FUNCTION;
    return RV32I_BC_AUIPC;
  case RV32I_JAL: return RV32I_BC_JAL;
  case RV32I_JALR: return RV32I_BC_JALR;
  case RV32I_OP_IMM:
    if (instr.Itype.rd == 0) return RV32I_BC_FUNCTION;
    switch (instr.Itype.funct3) {
    case 0x0:
      if (instr.Itype.rs1 == 0) return RV32I_BC_LI;
      else if (instr.Itype.signed_imm() == 0) return RV32I_BC_MV;
      else return RV32I_BC_ADDI;
    case 0x1: // SLLI, ...
      if (instr.Itype.high_bits() == 0x0) return RV32I_BC_SLLI;
      else if (instr.Itype.imm == 0b011000000100) // SEXT.B
        return RV32I_BC_SEXT_B;
      else if (instr.Itype.imm == 0b011000000101) // SEXT.H
        return RV32I_BC_SEXT_H;
      else if (instr.Itype.high_bits() == 0x280) // BSETI
        return RV32I_BC_BSETI;
      else return RV32I_BC_FUNCTION;
    case 0x2: // SLTI
      return RV32I_BC_SLTI;
    case 0x3: // SLTIU
      return RV32I_BC_SLTIU;
    case 0x4: // XORI
      return RV32I_BC_XORI;
    case 0x5:
      if (instr.Itype.high_bits() == 0x0) return RV32I_BC_SRLI;
      else if (instr.Itype.is_srai()) return RV32I_BC_SRAI;
      else if (instr.Itype.high_bits() == 0x480) // BEXTI
        return RV32I_BC_BEXTI;
      else return RV32I_BC_FUNCTION;
    case 0x6: return RV32I_BC_ORI;
    case 0x7: return RV32I_BC_ANDI;
    default: return RV32I_BC_FUNCTION;
    }
  case RV32I_OP:
    if (instr.Itype.rd == 0) return RV32I_BC_FUNCTION;
    switch (instr.Rtype.jumptable_friendly_op()) {
    case 0x0: return RV32I_BC_OP_ADD;
    case 0x200: return RV32I_BC_OP_SUB;
    case 0x1: return RV32I_BC_OP_SLL;
    case 0x2: return RV32I_BC_OP_SLT;
    case 0x3: return RV32I_BC_OP_SLTU;
    case 0x4: return RV32I_BC_OP_XOR;
    case 0x5: return RV32I_BC_OP_SRL;
    case 0x6: return RV32I_BC_OP_OR;
    case 0x7: return RV32I_BC_OP_AND;
    case 0x10: return RV32I_BC_OP_MUL;
    case 0x14: return RV32I_BC_OP_DIV;
    case 0x15: return RV32I_BC_OP_DIVU;
    case 0x16: return RV32I_BC_OP_REM;
    case 0x17: return RV32I_BC_OP_REMU;
    case 0x44: // ZEXT.H
      return RV32I_BC_OP_ZEXT_H;
    case 0x102: return RV32I_BC_OP_SH1ADD;
    case 0x104: return RV32I_BC_OP_SH2ADD;
    case 0x106: return RV32I_BC_OP_SH3ADD;
    case 0x205: return RV32I_BC_OP_SRA;
    case 0x141: // BSET
    case 0x142: // BCLR
    case 0x143: // BINV
    case 0x204: // XNOR
    case 0x206: // ORN
    case 0x207: // ANDN
    case 0x245: // BEXT
    case 0x54:  // MIN
    case 0x55:  // MINU
    case 0x56:  // MAX
    case 0x57:  // MAXU
    case 0x301: // ROL
    case 0x305: // ROR
    default: return RV32I_BC_FUNCTION;
    }
#ifdef RISCV_64I
  case RV64I_OP32:
    if constexpr (W < 8) return RV32I_BC_INVALID;

    switch (instr.Rtype.jumptable_friendly_op()) {
    case 0x0: // ADD.W
      return RV64I_BC_OP_ADDW;
    case 0x200: // SUB.W
      return RV64I_BC_OP_SUBW;
    case 0x10: // MUL.W
      return RV64I_BC_OP_MULW;
    case 0x40: // ADD.UW
      return RV64I_BC_OP_ADD_UW;
    case 0x44: // ZEXT.H
      return RV32I_BC_OP_ZEXT_H;
    default: return RV32I_BC_FUNCTION;
    }
  case RV64I_OP_IMM32:
    if constexpr (W < 8) return RV32I_BC_INVALID;

    if (instr.Itype.rd == 0) return RV32I_BC_FUNCTION;
    switch (instr.Itype.funct3) {
    case 0x0: return RV64I_BC_ADDIW;
    case 0x1: // SLLIW
      if (instr.Itype.high_bits() == 0x000) {
        return RV64I_BC_SLLIW;
      }
      return RV32I_BC_FUNCTION;
    case 0x5: // SRLIW / SRAIW
      if (instr.Itype.high_bits() == 0x000) {
        return RV64I_BC_SRLIW;
      }
      return RV32I_BC_FUNCTION;
    }
    return RV32I_BC_FUNCTION;
#endif
  case RV32I_SYSTEM:
    if (LIKELY(instr.Itype.funct3 == 0)) {
      if (instr.Itype.imm == 0) {
        return RV32I_BC_SYSCALL;
      }
      // WFI and STOP
      if (instr.Itype.imm == 0x105 || instr.Itype.imm == 0x7ff) {
        return RV32I_BC_STOP;
      }
    }
    return RV32I_BC_SYSTEM;
  case RV32I_FENCE: return RV32I_BC_FUNCTION;
  case RV32F_LOAD: {
    const rv32f_instruction fi{instr};
    switch (fi.Itype.funct3) {
    case 0x2: // FLW
      return RV32F_BC_FLW;
    case 0x3: // FLD
      return RV32F_BC_FLD;
#ifdef RISCV_EXT_VECTOR
    case 0x6: // VLE32
      return RV32V_BC_VLE32;
#endif
    default: return RV32I_BC_INVALID;
    }
  }
  case RV32F_STORE: {
    const rv32f_instruction fi{instr};
    switch (fi.Itype.funct3) {
    case 0x2: // FSW
      return RV32F_BC_FSW;
    case 0x3: // FSD
      return RV32F_BC_FSD;
#ifdef RISCV_EXT_VECTOR
    case 0x6: // VSE32
      return RV32V_BC_VSE32;
#endif
    default: return RV32I_BC_INVALID;
    }
  }
  case RV32F_FMADD: return RV32F_BC_FMADD;
  case RV32F_FMSUB:
  case RV32F_FNMADD:
  case RV32F_FNMSUB: return RV32I_BC_FUNCTION;
  case RV32F_FPFUNC:
    if (rv32f_instruction{instr}.R4type.funct2 >= 2) return RV32I_BC_FUNCTION;
    switch (instr.fpfunc()) {
    case 0b00000: // FADD
      return RV32F_BC_FADD;
    case 0b00001: // FSUB
      return RV32F_BC_FSUB;
    case 0b00010: // FMUL
      return RV32F_BC_FMUL;
    case 0b00011: // FDIV
      return RV32F_BC_FDIV;
    default: return RV32I_BC_FUNCTION;
    }
#ifdef RISCV_EXT_VECTOR
  case RV32V_OP: {
    const rv32v_instruction vi{instr};
    switch (instr.vwidth()) {
    case 0x1: // OPF.VV
      switch (vi.OPVV.funct6) {
      case 0b000000: // VFADD.VV
        return RV32V_BC_VFADD_VV;
      }
      break;
    case 0x5: // OPF.VF
      switch (vi.OPVV.funct6) {
      case 0b100100: // VFMUL.VF
        return RV32V_BC_VFMUL_VF;
      }
      break;
    }
    return RV32I_BC_FUNCTION;
  }
#endif
#ifdef RISCV_EXT_ATOMICS
  case RV32A_ATOMIC: return RV32I_BC_FUNCTION;
#endif
  }
  // Unknown instructions can be custom-handled
  return RV32I_BC_FUNCTION;
} // computed_index_for()

// rv32i/rv64i.cpp
template <int W> RISCV_INTERNAL const CPU<W>::instruction_t &CPU<W>::decode(const format_t instruction) {
  return decode_one<W>(instruction);
}

template <int W> RISCV_INTERNAL void CPU<W>::execute(const format_t instruction) {
  auto dec = decode(instruction);
  dec.handler(*this, instruction);
}

template <int W> RISCV_INTERNAL void CPU<W>::execute(uint8_t &handler_idx, uint32_t instr) {
  if (handler_idx == 0 && instr != 0) {
    [[unlikely]];
    handler_idx = DecoderData<W>::handler_index_for(decode(instr).handler);
  }
  DecoderData<W>::get_handlers()[handler_idx](*this, instr);
}

template <int W> const Instruction<W> &CPU<W>::get_unimplemented_instruction() noexcept {
  if constexpr (W == 4) return instr32i_UNIMPLEMENTED;
  else return instr64i_UNIMPLEMENTED;
}

template <int W> RISCV_COLD_PATH() std::string Registers<W>::to_string() const {
  char buffer[600];
  int len = 0;
  for (int i = 1; i < 32; i++) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%08X] ", RISCV::regname(i), this->get(i));
    if (i % 5 == 4) len += snprintf(buffer + len, sizeof(buffer) - len, "\n");
  }
  return std::string(buffer, len);
}

template <int W>
RISCV_COLD_PATH()
std::string CPU<W>::to_string(instruction_format format, const instruction_t &instr) const {
  char buffer[256];
  char ibuffer[128];
  int ibuflen = instr.printer(ibuffer, sizeof(ibuffer), *this, format);
  int len = 0;
  if (format.length() == 4) {
    len = snprintf(buffer, sizeof(buffer), "[%08X] %08X %.*s", this->pc(), format.whole, ibuflen, ibuffer);
  } else if (format.length() == 2) {
    len =
        snprintf(buffer, sizeof(buffer), "[%08X]     %04hX %.*s", this->pc(), (uint16_t)format.whole, ibuflen, ibuffer);
  } else {
    throw MachineException(UNIMPLEMENTED_INSTRUCTION_LENGTH, "Unimplemented instruction format length",
                           format.length());
  }
  return std::string(buffer, len);
}
} // namespace riscv
