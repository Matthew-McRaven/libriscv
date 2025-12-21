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

template <AddressType address_t> void Memory<address_t>::evict_execute_segments() {
  // destructor could throw, so let's invalidate early
  machine().cpu.set_execute_segment(*CPU<address_t>::empty_execute_segment());

  auto &main_segment = m_main_exec_segment;
  if (main_segment) {
    const SegmentKey key = SegmentKey::from(*main_segment, memory_arena_size());
    main_segment = nullptr;
    shared_execute_segments<address_t>.remove_if_unique(key);
  }

  while (!m_exec.empty()) {
    try {
      auto &segment = m_exec.back();
      if (segment) {
        const SegmentKey key = SegmentKey::from(*segment, memory_arena_size());
        segment = nullptr;
        shared_execute_segments<address_t>.remove_if_unique(key);
      }
      m_exec.pop_back();
    } catch (...) {
      // Ignore exceptions
    }
  }
}

template <AddressType address_t> void Memory<address_t>::evict_execute_segment(DecodedExecuteSegment<address_t> &segment) {
  const SegmentKey key = SegmentKey::from(segment, memory_arena_size());
  for (auto &seg : m_exec) {
    if (seg.get() == &segment) {
      seg = nullptr;
      if (&seg == &m_exec.back()) m_exec.pop_back();
      break;
    }
  }
  shared_execute_segments<address_t>.remove_if_unique(key);
}

// An execute segment contains a sequential array of raw instruction bits
// belonging to a set of sequential pages with +exec permission.
// It also contains a decoder cache that is produced from this instruction data.
// It is not strictly necessary to store the raw instruction bits, however, it
// enables step by step simulation as well as CLI- and remote debugging without
// rebuilding the emulator.
// Crucially, because of page alignments and 4 extra bytes, the necessary checks
// when reading from the execute segment is reduced. You can always read 4 bytes
// no matter where you are in the segment, a whole instruction unchecked.
template <AddressType address_t>
RISCV_INTERNAL DecodedExecuteSegment<address_t> &
Memory<address_t>::create_execute_segment(const MachineOptions<address_t> &options, const void *vdata, address_t vaddr, size_t exlen,
                                  bool is_initial, bool is_likely_jit) {
  if (UNLIKELY(exlen % (compressed_enabled ? 2 : 4)))
    throw MachineException(INVALID_PROGRAM, "Misaligned execute segment length");

  constexpr address_t PMASK = Page::size() - 1;
  const address_t pbase = vaddr & ~PMASK;
  const size_t prelen = vaddr - pbase;
  // Make 4 bytes of extra room to avoid having to validate 4-byte reads
  // when reading at 2 bytes before the end of the execute segment.
  const size_t midlen = exlen + prelen + 2; // Extra room for reads
  const size_t plen = (midlen + PMASK) & ~PMASK;
  // Because postlen uses midlen, we end up zeroing the extra 4 bytes in the end
  const size_t postlen = plen - midlen;
  // printf("Addr 0x%X Len %zx becomes 0x%X->0x%X PRE %zx MIDDLE %zu POST %zu TOTAL %zu\n",
  //	vaddr, exlen, pbase, pbase + plen, prelen, exlen, postlen, plen);
  if (UNLIKELY(prelen > plen || prelen + exlen > plen)) {
    throw MachineException(INVALID_PROGRAM, "Segment virtual base was bogus");
  }
#ifdef _MSC_VER
  if (UNLIKELY(pbase + plen < pbase)) throw MachineException(INVALID_PROGRAM, "Segment virtual base was bogus");
#else
  [[maybe_unused]] address_t pbase2;
  if (UNLIKELY(__builtin_add_overflow(pbase, plen, &pbase2)))
    throw MachineException(INVALID_PROGRAM, "Segment virtual base was bogus");
#endif
  // Create the whole executable memory range
  auto current_exec = std::make_shared<DecodedExecuteSegment<address_t>>(pbase, plen, vaddr, exlen);

  auto *exec_data = current_exec->exec_data(pbase);
  // This is a zeroed prologue in order to be able to use whole pages
  std::memset(&exec_data[0], 0, prelen);
  // This is the actual instruction bytes
  std::memcpy(&exec_data[prelen], vdata, exlen);
  // This memset() operation will end up zeroing the extra 4 bytes
  std::memset(&exec_data[prelen + exlen], 0, postlen);

  // Create CRC32-C hash of the execute segment
  const uint32_t hash = crc32c(exec_data, current_exec->exec_end() - current_exec->exec_begin());

  // Get a free slot to reference the execute segment
  auto &free_slot = this->next_execute_segment();

  if (options.use_shared_execute_segments) {
    // We have to key on the base address of the execute segment as well as the hash
    const SegmentKey key{uint64_t(current_exec->exec_begin()), hash, memory_arena_size()};

    // In order to prevent others from creating the same execute segment
    // we need to lock the shared execute segments mutex.
    auto &segment = shared_execute_segments<address_t>.get_segment(key);
    std::scoped_lock lock(segment.mutex);

    if (segment.segment != nullptr) {
      free_slot = segment.segment;
      return *free_slot;
    }

    // We need to create a new execute segment, as there is no shared
    // execute segment with the same hash.
    free_slot = std::move(current_exec);
    free_slot->set_likely_jit(is_likely_jit);
#if defined(RISCV_BINARY_TRANSLATION) && defined(RISCV_DEBUG)
    free_slot->set_record_slowpaths(options.record_slowpaths_to_jump_hints && !is_likely_jit);
#endif
    // Store the hash in the decoder cache
    free_slot->set_crc32c_hash(hash);

    this->generate_decoder_cache(options, free_slot, is_initial);

    // Share the execute segment
    shared_execute_segments<address_t>.get_segment(key).unlocked_set(free_slot);
  } else {
    free_slot = std::move(current_exec);
    free_slot->set_likely_jit(is_likely_jit);
    // Store the hash in the decoder cache
    free_slot->set_crc32c_hash(hash);

    this->generate_decoder_cache(options, free_slot, is_initial);
  }

  return *free_slot;
}

template <AddressType address_t> std::shared_ptr<DecodedExecuteSegment<address_t>> &Memory<address_t>::next_execute_segment() {
  if (!m_main_exec_segment) {
    return m_main_exec_segment;
  }
  if (LIKELY(m_exec.size() < RISCV_MAX_EXECUTE_SEGS)) {
    m_exec.push_back(nullptr);
    return m_exec.back();
  }
  throw MachineException(INVALID_PROGRAM, "Max execute segments reached");
}

template <AddressType address_t> const std::shared_ptr<DecodedExecuteSegment<address_t>> &Memory<address_t>::exec_segment_for(address_t vaddr) const {
  return const_cast<Memory<address_t> *>(this)->exec_segment_for(vaddr);
}

// The decoder cache is a sequential array of DecoderData<address_t> entries
// each of which (currently) serves a dual purpose of enabling
// threaded dispatch (m_bytecode) and fallback to callback function
// (m_handler). This enables high-speed emulation, precise simulation,
// CLI debugging and remote GDB debugging without rebuilding the emulator.
//
// The decoder cache covers all pages that the execute segment belongs
// in, so that all legal jumps (based on page +exec permission) will
// result in correct execution (including invalid instructions).
//
// The goal of the decoder cache is to allow uninterrupted execution
// with minimal bounds-checking, while also enabling accurate
// instruction counting.
template <AddressType address_t>
RISCV_INTERNAL void Memory<address_t>::generate_decoder_cache([[maybe_unused]] const MachineOptions<address_t> &options,
                                                      std::shared_ptr<DecodedExecuteSegment<address_t>> &shared_segment,
                                                      [[maybe_unused]] bool is_initial) {
  TIME_POINT(t0);
  auto &exec = *shared_segment;
  if (exec.exec_end() < exec.exec_begin()) throw MachineException(INVALID_PROGRAM, "Execute segment was invalid");

  const auto pbase = exec.pagedata_base();
  const auto addr = exec.exec_begin();
  const auto len = exec.exec_end() - exec.exec_begin();
  constexpr size_t PMASK = Page::size() - 1;
  // We need to allocate room for at least one more decoder cache entry.
  // This is because jump and branch instructions don't check PC after
  // not branching. The last entry is an invalid instruction.
  const size_t prelen = addr - pbase;
  const size_t midlen = len + prelen + 4; // Extra entry
  const size_t plen = (midlen + PMASK) & ~PMASK;
  // printf("generate_decoder_cache: Addr 0x%X Len %zx becomes 0x%X->0x%X PRE %zx MIDDLE %zu TOTAL %zu\n",
  //	addr, len, pbase, pbase + plen, prelen, midlen, plen);

  const size_t n_pages = plen / Page::size();
  if (n_pages == 0) {
    throw MachineException(INVALID_PROGRAM, "Program produced empty decoder cache");
  }
  // Here we allocate the decoder cache which is page-sized
  auto *decoder_cache = exec.create_decoder_cache(new DecoderCache<address_t>[n_pages], n_pages);
  // Clear the decoder cache! (technically only needed when binary translation is enabled)
  std::memset(decoder_cache, 0, n_pages * sizeof(DecoderCache<address_t>));
  // Get a base address relative pointer to the decoder cache
  // Eg. exec_decoder[pbase] is the first entry in the decoder cache
  // so that PC with a simple shift can be used as a direct index.
  auto *exec_decoder = decoder_cache[0].get_base() - pbase / DecoderCache<address_t>::DIVISOR;
  exec.set_decoder(exec_decoder);

  DecoderData<address_t> invalid_op;
  invalid_op.set_handler(this->machine().cpu.decode({0}));
  if (UNLIKELY(invalid_op.m_handler != 0)) {
    throw MachineException(INVALID_PROGRAM, "The invalid instruction did not have the index zero",
                           invalid_op.m_handler);
  }

  // PC-relative pointer to instruction bits
  auto *exec_segment = exec.exec_data();
  TIME_POINT(t1);

  // When compressed instructions are enabled, many decoder
  // entries are illegal because they are between instructions.
  bool was_full_instruction = true;

  /* Generate all instruction pointers for executable code.
     Cannot step outside of this area when pregen is enabled,
     so it's fine to leave the boundries alone. */
  TIME_POINT(t2);
  address_t dst = addr;
  const address_t end_addr = addr + len;
  for (; dst < addr + len;) {
    auto &entry = exec_decoder[dst / DecoderCache<address_t>::DIVISOR];
    entry.m_handler = 0;
    entry.idxend = 0;

    // Load unaligned instruction from execute segment
    const auto instruction = read_instruction(exec_segment, dst, end_addr);
    rv32i_instruction rewritten = instruction;

    if (!compressed_enabled || was_full_instruction) {
      // Cache the (modified) instruction bits
      auto bytecode = CPU<address_t>::computed_index_for(instruction);
      // Threaded rewrites are **always** enabled
      bytecode = exec.threaded_rewrite(bytecode, dst, rewritten);
      entry.set_bytecode(bytecode);
      entry.instr = rewritten.whole;
    } else {
      // WARNING: If we don't ignore this instruction,
      // it will get *wrong* idxend values, and cause *invalid jumps*
      entry.m_handler = 0;
      entry.set_bytecode(0);
      // ^ Must be made invalid, even if technically possible to jump to!
    }
    if constexpr (VERBOSE_DECODER) {
      if (entry.get_bytecode() >= RV32I_BC_BEQ && entry.get_bytecode() <= RV32I_BC_BGEU) {
        fprintf(stderr, "Detected branch bytecode at 0x%lX\n", dst);
      }
      if (entry.get_bytecode() == RV32I_BC_BEQ_FW || entry.get_bytecode() == RV32I_BC_BNE_FW) {
        fprintf(stderr, "Detected forward branch bytecode at 0x%lX\n", dst);
      }
    }

    // Increment PC after everything
    if constexpr (compressed_enabled) {
      // With compressed we always step forward 2 bytes at a time
      dst += 2;
      if (was_full_instruction) {
        // For it to be a full instruction again,
        // the length needs to match.
        was_full_instruction = (instruction.length() == 2);
      } else {
        // If it wasn't a full instruction last time, it
        // will for sure be one now.
        was_full_instruction = true;
      }
    } else dst += 4;
  }
  // Make sure the last entry is an invalid instruction
  // This simplifies many other sub-systems
  auto &entry = exec_decoder[(addr + len) / DecoderCache<address_t>::DIVISOR];
  entry.set_bytecode(0);
  entry.m_handler = 0;
  entry.idxend = 0;
  TIME_POINT(t3);

  realize_fastsim<address_t>(addr, dst, exec_segment, exec_decoder);

  // Debugging: EBREAK locations
  for (auto &loc : options.ebreak_locations) {
    address_t addr = 0;
    if (std::holds_alternative<address_t>(loc)) addr = std::get<address_t>(loc);
    else addr = machine().address_of(std::get<std::string>(loc));

    if (addr != 0x0 && addr >= exec.exec_begin() && addr < exec.exec_end()) {
      CPU<address_t>::install_ebreak_for(exec, addr);
      if (options.verbose_loader) {
        printf("libriscv: Added ebreak location at 0x%" PRIx64 "\n", uint64_t(addr));
      }
    }
  }

  TIME_POINT(t4);
#ifdef ENABLE_TIMINGS
  const long t1t0 = nanodiff(t0, t1);
  const long t2t1 = nanodiff(t1, t2);
  const long t3t2 = nanodiff(t2, t3);
  const long t3t4 = nanodiff(t3, t4);
  printf("libriscv: Decoder cache allocation took %ld ns\n", t1t0);
  if constexpr (binary_translation_enabled) printf("libriscv: Decoder cache bintr activation took %ld ns\n", t2t1);
  printf("libriscv: Decoder cache generation took %ld ns\n", t3t2);
  printf("libriscv: Decoder cache realization took %ld ns\n", t3t4);
  printf("libriscv: Decoder cache totals: %ld us\n", nanodiff(t0, t4) / 1000);
#endif
}

} // namespace riscv
