#include "cpu.hpp"

#include "instr_helpers.hpp"
#include "machine.hpp"

#if __has_include(<atomic>)
#define USE_ATOMIC_OPS __cpp_lib_atomic_ref
#include <atomic>
#else
#define USE_ATOMIC_OPS 0
#endif
#include <cstdint>
#include <inttypes.h>
static const char atomic_type[] { '?', '?', 'W', 'D', 'Q', '?', '?', '?' };
static const char* atomic_name2[] {
	"AMOADD", "AMOXOR", "AMOOR", "AMOAND", "AMOMIN", "AMOMAX", "AMOMINU", "AMOMAXU"
};

namespace riscv
{
	template <int W>
	template <typename Type>
	inline void CPU<W>::amo(format_t instr,
		Type(*op)(CPU&, Type&, uint32_t))
	{
		// 1. load address from rs1
		const auto addr = this->reg(instr.Atype.rs1);
		// 2. verify address alignment vs Type
		if (UNLIKELY(addr % sizeof(Type) != 0)) {
			trigger_exception(INVALID_ALIGNMENT, addr);
		}
		// 3. read value from writable memory location
		// TODO: Make Type unsigned to match other templates, avoiding spam
		Type& mem = machine().memory.template writable_read<Type> (addr);
		// 4. apply <op>, writing the value to mem and returning old value
		const Type old_value = op(*this, mem, instr.Atype.rs2);
		// 5. place value into rd
		// NOTE: we have to do it in this order, because we can
		// clobber rs2 when writing to rd, if they are the same!
		if (instr.Atype.rd != 0) {
			// For RV64, 32-bit AMOs always sign-extend the value
			// placed in rd, and ignore the upper 32 bits of the original
			// value of rs2.
			using signed_t = std::make_signed_t<Type>;
			this->reg(instr.Atype.rd) = (RVSIGNTYPE(*this))signed_t(old_value);
		}
	}

  const auto AMOADD_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_add(cpu.reg(rs2));
#else
			auto old_value = value;
			value += cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOADD_W_printer = [](char *buffer, size_t len, auto &, rv32i_instruction instr) RVPRINTR_ATTR {
    return snprintf(buffer, len, "%s.%c [%s] %s, %s", atomic_name2[instr.Atype.funct5 >> 2],
                    atomic_type[instr.Atype.funct3 & 7], RISCV::regname(instr.Atype.rs1),
                    RISCV::regname(instr.Atype.rs2), RISCV::regname(instr.Atype.rd));
  };
  const auto AMOXOR_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_xor(cpu.reg(rs2));
#else
			auto old_value = value;
			value ^= cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOOR_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_or(cpu.reg(rs2));
#else
			auto old_value = value;
			value |= cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOAND_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_and(cpu.reg(rs2));
#else
			auto old_value = value;
			value &= cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOMAX_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
      auto old_val = value;
      value = std::max(value, (int32_t)cpu.reg(rs2));
      return old_val;
    });
  };
  const auto AMOMIN_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
      auto old_val = value;
      value = std::min(value, (int32_t)cpu.reg(rs2));
      return old_val;
    });
  };
  const auto AMOMAXU_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<uint32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
      auto old_val = value;
      value = std::max(value, (uint32_t)cpu.reg(rs2));
      return old_val;
    });
  };
  const auto AMOMINU_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<uint32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
      auto old_val = value;
      value = std::min(value, (uint32_t)cpu.reg(rs2));
      return old_val;
    });
  };
  const auto AMOADD_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_add(cpu.reg(rs2));
#else
			auto old_value = value;
			value += cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOXOR_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_xor(cpu.reg(rs2));
#else
			auto old_value = value;
			value ^= cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOOR_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_or(cpu.reg(rs2));
#else
			auto old_value = value;
			value |= cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOAND_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_and(cpu.reg(rs2));
#else
			auto old_value = value;
			value &= cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOMAX_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
      auto old_val = value;
      value = std::max(value, int64_t(cpu.reg(rs2)));
      return old_val;
    });
  };
  const auto AMOMIN_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
      auto old_val = value;
      value = std::min(value, int64_t(cpu.reg(rs2)));
      return old_val;
    });
  };

  const auto AMOMAXU_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<uint64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
      auto old_val = value;
      value = std::max(value, (uint64_t)cpu.reg(rs2));
      return old_val;
    });
  };
  const auto AMOMINU_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<uint64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
      auto old_val = value;
      value = std::min(value, (uint64_t)cpu.reg(rs2));
      return old_val;
    });
  };

  const auto AMOSWAP_W_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int32_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).exchange(cpu.reg(rs2));
#else
			auto old_value = value;
			value = cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto AMOSWAP_W_printer = [](char *buffer, size_t len, auto &, rv32i_instruction instr) RVPRINTR_ATTR {
    return snprintf(buffer, len, "AMOSWAP.%c [%s] %s, %s", atomic_type[instr.Atype.funct3 & 7],
                    RISCV::regname(instr.Atype.rs1), RISCV::regname(instr.Atype.rs2), RISCV::regname(instr.Atype.rd));
  };

  const auto AMOSWAP_D_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    cpu.template amo<int64_t>(instr, [](auto &cpu, auto &value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).exchange(cpu.reg(rs2));
#else
			auto old_value = value;
			value = cpu.reg(rs2);
			return old_value;
#endif
    });
  };
  const auto LOAD_RESV_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    const auto addr = cpu.reg(instr.Atype.rs1);
    RVSIGNTYPE(cpu) value;
    // switch on atomic type
    if (instr.Atype.funct3 == AMOSIZE_W) {
      if (!cpu.atomics().load_reserve(4, addr)) cpu.trigger_exception(DEADLOCK_REACHED);
      value = (int32_t)cpu.machine().memory.template read<uint32_t>(addr);
    } else if (instr.Atype.funct3 == AMOSIZE_D) {
      if constexpr (RVISGE64BIT(cpu)) {
        if (!cpu.atomics().load_reserve(8, addr)) cpu.trigger_exception(DEADLOCK_REACHED);
        value = (int64_t)cpu.machine().memory.template read<uint64_t>(addr);
      } else cpu.trigger_exception(ILLEGAL_OPCODE);
    } else if (instr.Atype.funct3 == AMOSIZE_Q) {
      if constexpr (RVIS128BIT(cpu)) {
        if (!cpu.atomics().load_reserve(16, addr)) cpu.trigger_exception(DEADLOCK_REACHED);
        value = cpu.machine().memory.template read<RVREGTYPE(cpu)>(addr);
      } else cpu.trigger_exception(ILLEGAL_OPCODE);
    } else {
      cpu.trigger_exception(ILLEGAL_OPCODE);
    }
    if (instr.Atype.rd != 0) cpu.reg(instr.Atype.rd) = value;
  };

  const auto LOAD_RESV_printer = [](char *buffer, size_t len, auto &cpu, rv32i_instruction instr) RVPRINTR_ATTR {
    const uint64_t addr = cpu.reg(instr.Atype.rs1);
    return snprintf(buffer, len, "LR.%c [%s = 0x%" PRIX64 "], %s", atomic_type[instr.Atype.funct3 & 7],
                    RISCV::regname(instr.Atype.rs1), addr, RISCV::regname(instr.Atype.rd));
  };
  const auto STORE_COND_handler = [](auto &cpu, rv32i_instruction instr) RVINSTR_COLDATTR {
    const auto addr = cpu.reg(instr.Atype.rs1);
    bool resv = false;
    if (instr.Atype.funct3 == AMOSIZE_W) {
      resv = cpu.atomics().store_conditional(4, addr);
      if (resv) {
        cpu.machine().memory.template write<uint32_t>(addr, cpu.reg(instr.Atype.rs2));
      }
    } else if (instr.Atype.funct3 == AMOSIZE_D) {
      if constexpr (RVISGE64BIT(cpu)) {
        resv = cpu.atomics().store_conditional(8, addr);
        if (resv) {
          cpu.machine().memory.template write<uint64_t>(addr, cpu.reg(instr.Atype.rs2));
        }
      } else cpu.trigger_exception(ILLEGAL_OPCODE);
    } else if (instr.Atype.funct3 == AMOSIZE_Q) {
      if constexpr (RVIS128BIT(cpu)) {
        resv = cpu.atomics().store_conditional(16, addr);
        if (resv) {
          cpu.machine().memory.template write<RVREGTYPE(cpu)>(addr, cpu.reg(instr.Atype.rs2));
        }
      } else cpu.trigger_exception(ILLEGAL_OPCODE);
    } else {
      cpu.trigger_exception(ILLEGAL_OPCODE);
    }
    // Write non-zero value to RD on failure
    if (instr.Atype.rd != 0) cpu.reg(instr.Atype.rd) = !resv;
  };
  const auto STORE_COND_printer = [](char *buffer, size_t len, auto &, rv32i_instruction instr) RVPRINTR_ATTR {
    return snprintf(buffer, len, "SC.%c [%s], %s res=%s", atomic_type[instr.Atype.funct3 & 7],
                    RISCV::regname(instr.Atype.rs1), RISCV::regname(instr.Atype.rs2), RISCV::regname(instr.Atype.rd));
  };

  } // namespace riscv

#ifdef RISCV_32I
  const riscv::Instruction<4> instr32i_AMOADD_W{riscv::AMOADD_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOXOR_W{riscv::AMOXOR_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOOR_W{riscv::AMOOR_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOAND_W{riscv::AMOAND_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOMAX_W{riscv::AMOMAX_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOMIN_W{riscv::AMOMIN_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOMAXU_W{riscv::AMOMAXU_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOMINU_W{riscv::AMOMINU_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOADD_D{riscv::AMOADD_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOXOR_D{riscv::AMOXOR_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOOR_D{riscv::AMOOR_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOAND_D{riscv::AMOAND_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOMAX_D{riscv::AMOMAX_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOMIN_D{riscv::AMOMIN_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOMAXU_D{riscv::AMOMAXU_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOMINU_D{riscv::AMOMINU_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<4> instr32i_AMOSWAP_W{riscv::AMOSWAP_W_handler, riscv::AMOSWAP_W_printer};
  const riscv::Instruction<4> instr32i_AMOSWAP_D{riscv::AMOSWAP_D_handler, riscv::AMOSWAP_W_printer};
  const riscv::Instruction<4> instr32i_LOAD_RESV{riscv::LOAD_RESV_handler, riscv::LOAD_RESV_printer};
  const riscv::Instruction<4> instr32i_STORE_COND{riscv::STORE_COND_handler, riscv::STORE_COND_printer};

  const riscv::Instruction<8> instr64i_AMOADD_W{riscv::AMOADD_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOXOR_W{riscv::AMOXOR_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOOR_W{riscv::AMOOR_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOAND_W{riscv::AMOAND_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOMAX_W{riscv::AMOMAX_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOMIN_W{riscv::AMOMIN_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOMAXU_W{riscv::AMOMAXU_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOMINU_W{riscv::AMOMINU_W_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOADD_D{riscv::AMOADD_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOXOR_D{riscv::AMOXOR_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOOR_D{riscv::AMOOR_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOAND_D{riscv::AMOAND_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOMAX_D{riscv::AMOMAX_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOMIN_D{riscv::AMOMIN_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOMAXU_D{riscv::AMOMAXU_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOMINU_D{riscv::AMOMINU_D_handler, riscv::AMOADD_W_printer};
  const riscv::Instruction<8> instr64i_AMOSWAP_W{riscv::AMOSWAP_W_handler, riscv::AMOSWAP_W_printer};
  const riscv::Instruction<8> instr64i_AMOSWAP_D{riscv::AMOSWAP_D_handler, riscv::AMOSWAP_W_printer};
  const riscv::Instruction<8> instr64i_LOAD_RESV{riscv::LOAD_RESV_handler, riscv::LOAD_RESV_printer};
  const riscv::Instruction<8> instr64i_STORE_COND{riscv::STORE_COND_handler, riscv::STORE_COND_printer};
#endif
