#pragma once
#include "../../machine.hpp"
#include "../../memory/memory.hpp"
#include "decoded_exec_segment.hpp"
#include "decoder_cache.hpp"

#include <cstdint>
#include <mutex>
namespace riscv {

// cpu.cpp
// A default empty execute segment used to enforce that the
// current CPU execute segment is never null.
template <AddressType address_t> std::shared_ptr<DecodedExecuteSegment<address_t>> &CPU<address_t>::empty_execute_segment() noexcept {
  static std::shared_ptr<DecodedExecuteSegment<address_t>> empty_shared =
      std::make_shared<DecodedExecuteSegment<address_t>>(0, 0, 0, 0);
  return empty_shared;
}

template <AddressType address_t>
DecodedExecuteSegment<address_t> &CPU<address_t>::init_execute_area(const void *vdata, address_t begin, address_t vlength,
                                                    bool is_likely_jit) {
  if (vlength < 4) trigger_exception(EXECUTION_SPACE_PROTECTION_FAULT, begin);
  // Create a new *non-initial* execute segment
  if (machine().has_options())
    this->m_exec =
        &machine().memory.create_execute_segment(machine().options(), vdata, begin, vlength, false, is_likely_jit);
  else
    this->m_exec =
        &machine().memory.create_execute_segment(MachineOptions<address_t>(), vdata, begin, vlength, false, is_likely_jit);
  return *this->m_exec;
} // CPU::init_execute_area

template <AddressType address_t> DecoderData<address_t> &CPU<address_t>::create_block_ending_entry_at(DecodedExecuteSegment<address_t> &exec, address_t addr) {
  if (!exec.is_within(addr)) {
    throw MachineException(EXECUTION_SPACE_PROTECTION_FAULT, "Breakpoint address is not within the execute segment",
                           addr);
  }

  auto *exec_decoder = exec.decoder_cache();
  auto *decoder_begin = &exec_decoder[exec.exec_begin() / DecoderCache<address_t>::DIVISOR];

  auto &cache_entry = exec_decoder[addr / DecoderCache<address_t>::DIVISOR];

  // The last instruction will be the current entry
  // Later instructions will work as normal
  // 1. Look back to find the beginning of the block
  auto *last = &cache_entry;
  auto *current = &cache_entry;
  auto last_block_bytes = cache_entry.block_bytes();
  while (current > decoder_begin && (current - 1)->block_bytes() > last_block_bytes) {
    current--;
    last_block_bytes = current->block_bytes();
  }

  // 2. Find the start address of the block
  const auto block_begin_addr = addr - (compressed_enabled ? 2 : 4) * (last - current);
  if (!exec.is_within(block_begin_addr)) {
    throw MachineException(INVALID_PROGRAM, "Breakpoint block was outside execute area", block_begin_addr);
  }

  // 3. Correct block_bytes() for all entries in the block
  auto patched_addr = block_begin_addr;
  for (auto *dd = current; dd < last; dd++) {
    // Get the patched decoder entry
    auto &p = exec_decoder[patched_addr / DecoderCache<address_t>::DIVISOR];
    p.idxend = last - dd;
#ifdef RISCV_EXT_C
    p.icount = 0; // TODO: Implement C-ext icount for breakpoints
#endif
    patched_addr += (compressed_enabled) ? 2 : 4;
  }
  // Check if the last address matches the breakpoint address
  if (patched_addr != addr) {
    throw MachineException(INVALID_PROGRAM, "Last instruction in breakpoint block was not aligned", patched_addr);
  }

  return cache_entry;
}

// Install an ebreak instruction at the given address
template <AddressType address_t> uint32_t CPU<address_t>::install_ebreak_for(DecodedExecuteSegment<address_t> &exec, address_t breakpoint_addr) {
  // Get a reference to the decoder cache
  auto &cache_entry = CPU<address_t>::create_block_ending_entry_at(exec, breakpoint_addr);
  const auto old_instruction = cache_entry.instr;

  // Install the new ebreak instruction at the breakpoint address
  rv32i_instruction new_instruction;
  new_instruction.Itype.opcode = 0b1110011; // SYSTEM
  new_instruction.Itype.rd = 0;
  new_instruction.Itype.funct3 = 0b000;
  new_instruction.Itype.rs1 = 0;
  new_instruction.Itype.imm = 1; // EBREAK
  cache_entry.instr = new_instruction.whole;
  cache_entry.set_bytecode(RV32I_BC_SYSTEM);
  cache_entry.idxend = 0;
#ifdef RISCV_EXT_C
  cache_entry.icount = 0; // TODO: Implement C-ext icount for breakpoints
#endif

  // Return the old instruction
  return old_instruction;
}

template <AddressType address_t> uint32_t CPU<address_t>::install_ebreak_at(address_t addr) { return install_ebreak_for(*m_exec, addr); }

template <AddressType address_t> bool CPU<address_t>::create_fast_path_function(DecodedExecuteSegment<address_t> &exec, address_t block_pc) {
  // First, find the end of the block that either returns or stops (ignore traps)
  // 1. Return: JALR reg
  // 2. Stop: STOP
  if (!exec.is_within(block_pc)) {
    throw MachineException(EXECUTION_SPACE_PROTECTION_FAULT, "Function start address is not within the execute segment",
                           block_pc);
  }

  auto *exec_decoder = exec.decoder_cache();
  // The beginning of the function:
  auto *cache_entry = &exec_decoder[block_pc / DecoderCache<address_t>::DIVISOR];

  const address_t current_end = exec.exec_end();
  while (block_pc < current_end) {
    // Move to the end of the block
    block_pc += cache_entry->block_bytes();
    cache_entry += cache_entry->block_bytes() / DecoderCache<address_t>::DIVISOR;
    // Check if we're still within the execute segment
    if (UNLIKELY(block_pc >= current_end)) {
      // TODO: Return false instead?
      throw MachineException(INVALID_PROGRAM, "Function block ended outside execute area", block_pc);
    }
    // Check if we're at the end of the function
    auto bytecode = cache_entry->get_bytecode();
    if (bytecode == RV32I_BC_JALR) {
      const FasterItype instr{cache_entry->instr};

      // Check if it's a direct jump to REG_RA
      if (instr.rs2 == REG_RA && instr.rs1 == 0 && instr.imm == 0) {
        if (cache_entry->block_bytes() != 0)
          throw MachineException(INVALID_PROGRAM, "Function block ended but was not last instruction in block",
                                 block_pc);
        // We found the (potential) end of the function
        // Now rewrite it to a speculative live-patch STOP instruction
        cache_entry->set_atomic_bytecode_and_handler(RV32I_BC_LIVEPATCH, 1);
        return true;
      } else {
        // Unconditional jump could be a tail call, in which
        // case we can't confidently optimize this function
        return false;
      }
    } else if (bytecode == RV32I_BC_STOP) {
      // It's already a fast-path function
      return true;
    } else if (bytecode == RV32I_BC_LIVEPATCH) {
      // It's already (potentially) a fast-path function
      if (cache_entry->m_handler == 1 || cache_entry->m_handler == 2) {
        return true;
      }
#ifdef RISCV_EXT_COMPRESSED
    } else if (bytecode == RV32C_BC_JR) {
      const auto reg = cache_entry->instr;
      if (reg == REG_RA) {
        if (cache_entry->block_bytes() != 0)
          throw MachineException(INVALID_PROGRAM, "Function block ended but was not last instruction in block",
                                 block_pc);
        // We found the (potential) end of the function
        // Now rewrite it to a speculative live-patch STOP instruction
        cache_entry->set_atomic_bytecode_and_handler(RV32I_BC_LIVEPATCH, 2);
        return true;
      } else {
        return false;
      }
#endif
    }

    cache_entry++;
    block_pc += (compressed_enabled) ? 2 : 4;
  }
  // Not able to find the end of the function
  return false;
}

template <AddressType address_t> bool CPU<address_t>::create_fast_path_function(address_t addr) {
  DecodedExecuteSegment<address_t> *exec = machine().memory.exec_segment_for(addr).get();
  return create_fast_path_function(*exec, addr);
}

// decoder_cache.cpp

template <AddressType address_t> static SharedExecuteSegments<address_t> shared_execute_segments;

template <AddressType address_t> static bool is_regular_compressed(uint16_t instr) {
  const rv32c_instruction ci{instr};
#define CI_CODE(x, y) ((x << 13) | (y))
  switch (ci.opcode()) {
  case CI_CODE(0b001, 0b01):
    if constexpr (sizeof(address_t) == 8) return true; // C.ADDIW
    return false;                      // C.JAL 32-bit
  case CI_CODE(0b101, 0b01):           // C.JMP
  case CI_CODE(0b110, 0b01):           // C.BEQZ
  case CI_CODE(0b111, 0b01):           // C.BNEZ
    return false;
  case CI_CODE(0b100, 0b10): { // VARIOUS
    const bool topbit = ci.whole & (1 << 12);
    if (!topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0) {
      return false; // C.JR rd
    } else if (topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0) {
      return false; // C.JALR ra, rd+0
    } // TODO: Handle C.EBREAK
    return true;
  }
  default: return true;
  }
}

template <AddressType address_t>
static inline void fill_entries(const std::array<DecoderEntryAndCount<address_t>, 256> &block_array,
                                size_t block_array_count, address_t block_pc, address_t current_pc) {
  const unsigned last_count = block_array[block_array_count - 1].count;
  unsigned count = (current_pc - block_pc) >> 1;
  count -= last_count;
  if (count > 255) throw MachineException(INVALID_PROGRAM, "Too many non-branching instructions in a row");

  for (size_t i = 0; i < block_array_count; i++) {
    const DecoderEntryAndCount<address_t> &tuple = block_array[i];
    DecoderData<address_t> *entry = tuple.entry;
    const int length = tuple.count;

    // Ends at instruction *before* last PC
    entry->idxend = count;
    entry->icount = block_array_count - i;

    if constexpr (VERBOSE_DECODER) {
      fprintf(stderr, "Block 0x%lX has %u instructions\n", block_pc, count);
    }

    count -= length;
  }
}

template <AddressType address_t>
static void realize_fastsim(address_t base_pc, address_t last_pc, const uint8_t *exec_segment,
                            DecoderData<address_t> *exec_decoder) {
  if constexpr (compressed_enabled) {
    if (UNLIKELY(base_pc >= last_pc)) throw MachineException(INVALID_PROGRAM, "The execute segment has an overflow");
    if (UNLIKELY(base_pc & 0x1)) throw MachineException(INVALID_PROGRAM, "The execute segment is misaligned");

    // Go through entire executable segment and measure lengths
    // Record entries while looking for jumping instruction, then
    // fill out data and opcode lengths previous instructions.
    std::array<DecoderEntryAndCount<address_t>, 256> block_array;
    address_t pc = base_pc;
    while (pc < last_pc) {
      size_t block_array_count = 0;
      const address_t block_pc = pc;
      DecoderData<address_t> *entry = &exec_decoder[pc / DecoderCache<address_t>::DIVISOR];
      const AlignedLoad16 *iptr = (AlignedLoad16 *)&exec_segment[pc];
      const AlignedLoad16 *iptr_begin = iptr;
      while (true) {
        const unsigned length = iptr->length();
        const int count = length >> 1;

        // Record the instruction
        block_array[block_array_count++] = {entry, count};

        // Make sure PC does not overflow
#ifdef _MSC_VER
        if (pc + length < pc) throw MachineException(INVALID_PROGRAM, "PC overflow during execute segment decoding");
#else
        [[maybe_unused]] address_t pc2;
        if (UNLIKELY(__builtin_add_overflow(pc, length, &pc2)))
          throw MachineException(INVALID_PROGRAM, "PC overflow during execute segment decoding");
#endif
        pc += length;

        // If ending up crossing last_pc, it's an invalid block although
        // it could just be garbage, so let's force-end with an invalid instruction.
        if (UNLIKELY(pc > last_pc)) {
          entry->m_bytecode = 0; // Invalid instruction
          entry->m_handler = 0;
          break;
        }

        // All opcodes that can modify PC
        if (length == 2) {
          if (!is_regular_compressed<address_t>(iptr->half())) break;
        } else {
          const unsigned opcode = iptr->opcode();
          if (opcode == RV32I_BRANCH || opcode == RV32I_SYSTEM || opcode == RV32I_JAL || opcode == RV32I_JALR) break;
        }

        // A last test for the last instruction, which should have been a block-ending
        // instruction. Since it wasn't we must force-end the block here.
        if (UNLIKELY(pc >= last_pc)) {
          entry->m_bytecode = 0; // Invalid instruction
          entry->m_handler = 0;
          break;
        }

        iptr += count;

        // Too large blocks are likely malicious (although could be many empty pages)
        if (UNLIKELY(iptr - iptr_begin >= 255)) {
          // NOTE: Reinsert original instruction, as long sequences will lead to
          // PC becoming desynched, as it doesn't get increased.
          // We use a new block-ending fallback function handler instead.
          rv32i_instruction instruction = read_instruction(exec_segment, pc - length, last_pc);
          entry->set_bytecode(RV32I_BC_FUNCBLOCK);
          entry->set_invalid_handler(); // Resolve lazily
          entry->instr = instruction.whole;
          break;
        }

        entry += count;
      }
      if constexpr (VERBOSE_DECODER) {
        fprintf(stderr, "Block 0x%lX to 0x%lX\n", block_pc, pc);
      }

      if (UNLIKELY(block_array_count == 0))
        throw MachineException(INVALID_PROGRAM, "Encountered empty block after measuring");

      fill_entries(block_array, block_array_count, block_pc, pc);
    }
  } else { // !compressed_enabled
    // Count distance to next branching instruction backwards
    // and fill in idxend for all entries along the way.
    // This is for uncompressed instructions, which are always
    // 32-bits in size. We can use the idxend value for
    // instruction counting.
    unsigned idxend = 0;
    address_t pc = last_pc - 4;
    // NOTE: The last check avoids overflow
    while (pc >= base_pc && pc < last_pc) {
      const rv32i_instruction instruction = read_instruction(exec_segment, pc, last_pc);
      DecoderData<address_t> &entry = exec_decoder[pc / DecoderCache<address_t>::DIVISOR];
      const unsigned opcode = instruction.opcode();

      // All opcodes that can modify PC and stop the machine
      if (opcode == RV32I_BRANCH || opcode == RV32I_SYSTEM || opcode == RV32I_JAL || opcode == RV32I_JALR) idxend = 0;
      if (UNLIKELY(idxend == 65535)) {
        // It's a long sequence of instructions, so end block here.
        entry.set_bytecode(RV32I_BC_FUNCBLOCK);
        entry.set_invalid_handler(); // Resolve lazily
        entry.instr = instruction.whole;
        idxend = 0;
      }

      // Ends at *one instruction before* the block ends
      entry.idxend = idxend;
      // Increment after, idx becomes block count - 1
      idxend++;

      pc -= 4;
    }
  }
}

template <AddressType address_t> RISCV_INTERNAL size_t DecoderData<address_t>::handler_index_for(Handler new_handler) {
  std::scoped_lock lock(handler_idx_mutex);

  auto it = handler_cache.find(new_handler);
  if (it != handler_cache.end()) return it->second;

  if (UNLIKELY(handler_count >= instr_handlers.size()))
    throw MachineException(INVALID_PROGRAM, "Too many instruction handlers");
  instr_handlers[handler_count] = new_handler;
  const size_t idx = handler_count++;
  handler_cache.emplace(new_handler, idx);
  return idx;
}
} // namespace riscv
