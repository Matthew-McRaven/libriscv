#include "machine.hpp"

#include "decoder_cache.hpp"
#include "rv32i_instr.hpp"
#include "rvfd.hpp"

#define INSTRUCTION(x, ...) \
	static const CPU<4>::instruction_t instr32i_##x { __VA_ARGS__ }
#define DECODED_INSTR(x) instr32i_##x

#ifdef RISCV_EXT_ATOMICS
#include "rva.hpp"
#endif

#ifdef RISCV_EXT_VECTOR
#include "rvv_instr.cpp"
#endif
#include "instruction_list.hpp"

template <int W> const riscv::Instruction<W> &fuck_this(const riscv::instruction_format instruction) {
  // -*-C++-*-
  using address_t = riscv::address_type<W>;
  using namespace riscv;
#ifdef RISCV_EXT_COMPRESSED
  if (instruction.is_long()) // RV32 IMAFD
  {
#endif
    // Quadrant 3
    switch (instruction.opcode()) {
      // RV32IM
    case RV32I_LOAD:
      if (LIKELY(instruction.Itype.rd != 0)) {
        switch (instruction.Itype.funct3) {
        case 0x0: return DECODED_INSTR(LOAD_I8);
        case 0x1: return DECODED_INSTR(LOAD_I16);
        case 0x2: return DECODED_INSTR(LOAD_I32);
        case 0x3:
          if constexpr (sizeof(address_t) >= 8) {
            return (DECODED_INSTR(LOAD_I64));
          }
          return (DECODED_INSTR(ILLEGAL));
        case 0x4: return (DECODED_INSTR(LOAD_U8));
        case 0x5: return (DECODED_INSTR(LOAD_U16));
        case 0x6: return (DECODED_INSTR(LOAD_U32));
        case 0x7: [[fallthrough]];
        default: return (DECODED_INSTR(ILLEGAL));
        }
      } else {
        return (DECODED_INSTR(LOAD_X_DUMMY));
      }
      break;
    case RV32I_STORE:
      switch (instruction.Stype.funct3) {
      case 0x0:
        if (instruction.Stype.signed_imm() == 0) return (DECODED_INSTR(STORE_I8));
        return (DECODED_INSTR(STORE_I8_IMM));
      case 0x1: return (DECODED_INSTR(STORE_I16_IMM));
      case 0x2: return (DECODED_INSTR(STORE_I32_IMM));
      case 0x3:
        if constexpr (sizeof(address_t) >= 8) {
          return (DECODED_INSTR(STORE_I64_IMM));
        }
        return (DECODED_INSTR(ILLEGAL));
      case 0x4: [[fallthrough]];
      default: return (DECODED_INSTR(ILLEGAL));
      }
      break;
    case RV32I_BRANCH:
      switch (instruction.Btype.funct3) {
      case 0x0: return (DECODED_INSTR(BRANCH_EQ));
      case 0x1: return (DECODED_INSTR(BRANCH_NE));
      case 0x2:
      case 0x3: return (DECODED_INSTR(ILLEGAL));
      case 0x4: return (DECODED_INSTR(BRANCH_LT));
      case 0x5: return (DECODED_INSTR(BRANCH_GE));
      case 0x6: return (DECODED_INSTR(BRANCH_LTU));
      case 0x7: return (DECODED_INSTR(BRANCH_GEU));
      }
    case RV32I_JALR: return (DECODED_INSTR(JALR));
    case RV32I_JAL:
      if (instruction.Jtype.rd != 0) {
        return (DECODED_INSTR(JAL));
      } else {
        return (DECODED_INSTR(JMPI));
      }
    case RV32I_OP_IMM:
      if (LIKELY(instruction.Itype.rd != 0)) {
        switch (instruction.Itype.funct3) {
        case 0x0: // ADDI
          if (instruction.Itype.rs1 == 0) {
            return (DECODED_INSTR(OP_IMM_LI));
          } else if (instruction.Itype.imm == 0) {
            return (DECODED_INSTR(OP_MV));
          }
          return (DECODED_INSTR(OP_IMM_ADDI));
        case 0x1: // SLLI
          if (instruction.Itype.high_bits() == 0x0) {
            return (DECODED_INSTR(OP_IMM_SLLI));
          }
          return (DECODED_INSTR(OP_IMM));
        case 0x5: // SRLI / SRAI
          if (instruction.Itype.high_bits() == 0x0) {
            return (DECODED_INSTR(OP_IMM_SRLI));
          } else {
            return (DECODED_INSTR(OP_IMM));
          }
        case 0x7: // ANDI
          return (DECODED_INSTR(OP_IMM_ANDI));
        default: return (DECODED_INSTR(OP_IMM));
        }
      }
      return (DECODED_INSTR(NOP));
    case RV32I_OP:
      if (LIKELY(instruction.Rtype.rd != 0)) {
        switch (instruction.Rtype.jumptable_friendly_op()) {
        case 0x0: return (DECODED_INSTR(OP_ADD));
        case 0x200: return (DECODED_INSTR(OP_SUB));
        default: return (DECODED_INSTR(OP));
        }
      }
      return (DECODED_INSTR(NOP));
    case RV32I_SYSTEM:
      if (LIKELY(instruction.Itype.funct3 == 0)) {
        if (instruction.Itype.imm == 0) {
          return (DECODED_INSTR(SYSCALL));
        } else if (instruction.Itype.imm == 0x7FF) {
          return (DECODED_INSTR(WFI)); // STOP
        } else if (instruction.Itype.imm == 261) {
          return (DECODED_INSTR(WFI));
        }
      }
      return (DECODED_INSTR(SYSTEM));
    case RV32I_LUI:
      if (LIKELY(instruction.Utype.rd != 0)) {
        return (DECODED_INSTR(LUI));
      } else {
        return (DECODED_INSTR(NOP));
      }
    case RV32I_AUIPC:
      if (LIKELY(instruction.Utype.rd != 0)) {
        return instr32i_AUIPC;
      } else {
        return (DECODED_INSTR(NOP));
      }
    case RV64I_OP_IMM32:
      if (LIKELY(instruction.Itype.rd != 0)) {
        switch (instruction.Itype.funct3) {
        case 0x0: // ADDIW
          return (DECODED_INSTR(OP_IMM32_ADDIW));
        case 0x1: // SLLIW
          if (instruction.Itype.high_bits() == 0x000) {
            return (DECODED_INSTR(OP_IMM32_SLLIW));
          } else if (instruction.Itype.high_bits() == 0x080) {
            return (DECODED_INSTR(OP_IMM32_SLLI_UW));
          } else {
            return (DECODED_INSTR(OP_IMM32));
          }
        case 0x5: // SRLIW / SRAIW
          if (instruction.Itype.high_bits() == 0x000) {
            return (DECODED_INSTR(OP_IMM32_SRLIW));
          } else if (instruction.Itype.high_bits() == 0x400) {
            return (DECODED_INSTR(OP_IMM32_SRAIW));
          } else {
            return (DECODED_INSTR(OP_IMM32));
          }
          break;
        }
        return (DECODED_INSTR(ILLEGAL));
      } else {
        return (DECODED_INSTR(NOP));
      }
    case RV64I_OP32:
      if (LIKELY(instruction.Rtype.rd != 0)) {
        switch (instruction.Rtype.jumptable_friendly_op()) {
        case 0x0: // ADDW
          return (DECODED_INSTR(OP32_ADDW));
        default: return (DECODED_INSTR(OP32));
        }
      } else {
        return (DECODED_INSTR(NOP));
      }
    case RV32I_FENCE:
      return (DECODED_INSTR(FENCE));

      // RV32F & RV32D - Floating-point instructions
    case RV32F_LOAD: {
      const riscv::rv32f_instruction fi{instruction};
      switch (fi.Itype.funct3) {
      case 0x2: // FLW
        return (DECODED_INSTR(FLW));
      case 0x3: // FLD
        return (DECODED_INSTR(FLD));
#ifdef RISCV_EXT_VECTOR
      case 0x6: // VLE32
        return (DECODED_VECTOR(VLE32));
#endif
      default: return (DECODED_INSTR(ILLEGAL));
      }
    }
    case RV32F_STORE: {
      const rv32f_instruction fi{instruction};
      switch (fi.Itype.funct3) {
      case 0x2: // FSW
        return (DECODED_INSTR(FSW));
      case 0x3: // FSD
        return (DECODED_INSTR(FSD));
#ifdef RISCV_EXT_VECTOR
      case 0x6: // VSE32
        return (DECODED_VECTOR(VSE32));
#endif
      default: return (DECODED_INSTR(ILLEGAL));
      }
    }
    case RV32F_FMADD: return (DECODED_INSTR(FMADD));
    case RV32F_FMSUB: return (DECODED_INSTR(FMSUB));
    case RV32F_FNMSUB: return (DECODED_INSTR(FNMSUB));
    case RV32F_FNMADD: return (DECODED_INSTR(FNMADD));
    case RV32F_FPFUNC:
      switch (instruction.fpfunc()) {
      case 0b00000: return (DECODED_INSTR(FADD));
      case 0b00001: return (DECODED_INSTR(FSUB));
      case 0b00010: return (DECODED_INSTR(FMUL));
      case 0b00011: return (DECODED_INSTR(FDIV));
      case 0b00100: return (DECODED_INSTR(FSGNJ_NX));
      case 0b00101: return (DECODED_INSTR(FMIN_FMAX));
      case 0b01011: return (DECODED_INSTR(FSQRT));
      case 0b10100:
        if (rv32f_instruction{instruction}.R4type.rd != 0) return (DECODED_INSTR(FEQ_FLT_FLE));
        return (DECODED_INSTR(NOP));
      case 0b01000: return (DECODED_INSTR(FCVT_SD_DS));
      case 0b11000:
        if (rv32f_instruction{instruction}.R4type.rd != 0) return (DECODED_INSTR(FCVT_W_SD));
        return (DECODED_INSTR(NOP));
      case 0b11010: return (DECODED_INSTR(FCVT_SD_W));
      case 0b11100:
        if (rv32f_instruction{instruction}.R4type.rd != 0) {
          if (rv32f_instruction{instruction}.R4type.funct3 == 0) {
            return (DECODED_INSTR(FMV_X_W));
          } else {
            return (DECODED_INSTR(FCLASS));
          }
        }
        return (DECODED_INSTR(NOP));
      case 0b11110: return (DECODED_INSTR(FMV_W_X));
      }
      break;

#ifdef RISCV_EXT_VECTOR
    case RV32V_OP:
      switch (instruction.vwidth()) {
      case 0x0: // OPI.VV
        return (DECODED_VECTOR(VOPI_VV));
      case 0x1: // OPF.VV
        return (DECODED_VECTOR(VOPF_VV));
      case 0x2: // OPM.VV
        return (DECODED_VECTOR(VOPM_VV));
      case 0x3: // OPI.VI
        return (DECODED_VECTOR(VOPI_VI));
      case 0x5: // OPF.VF
        return (DECODED_VECTOR(VOPF_VF));
      case 0x7: // Vector Configuration
        switch (instruction.vsetfunc()) {
        case 0x0:
        case 0x1: return (DECODED_VECTOR(VSETVLI));
        case 0x2: return (DECODED_VECTOR(VSETVL));
        case 0x3: return (DECODED_VECTOR(VSETIVLI));
        }
      }
      break;
#endif
#ifdef RISCV_EXT_ATOMICS
      // RVxA - Atomic instructions
    case RV32A_ATOMIC:
      switch (instruction.Atype.funct3) {
      case AMOSIZE_W:
        switch (instruction.Atype.funct5) {
        case 0b00010:
          if (instruction.Atype.rs2 == 0) return (DECODED_INSTR(LOAD_RESV));
          return (DECODED_INSTR(ILLEGAL));
        case 0b00011: return (DECODED_INSTR(STORE_COND));
        case 0b00000: return (DECODED_INSTR(AMOADD_W));
        case 0b00001: return (DECODED_INSTR(AMOSWAP_W));
        case 0b00100: return (DECODED_INSTR(AMOXOR_W));
        case 0b01000: return (DECODED_INSTR(AMOOR_W));
        case 0b01100: return (DECODED_INSTR(AMOAND_W));
        case 0b10000: return (DECODED_INSTR(AMOMIN_W));
        case 0b10100: return (DECODED_INSTR(AMOMAX_W));
        case 0b11000: return (DECODED_INSTR(AMOMINU_W));
        case 0b11100: return (DECODED_INSTR(AMOMAXU_W));
        }
        break;
      case AMOSIZE_D:
        if constexpr (sizeof(address_t) >= 8) {
          switch (instruction.Atype.funct5) {
          case 0b00010:
            if (instruction.Atype.rs2 == 0) return (DECODED_INSTR(LOAD_RESV));
            return (DECODED_INSTR(ILLEGAL));
          case 0b00011: return (DECODED_INSTR(STORE_COND));
          case 0b00000: return (DECODED_INSTR(AMOADD_D));
          case 0b00001: return (DECODED_INSTR(AMOSWAP_D));
          case 0b00100: return (DECODED_INSTR(AMOXOR_D));
          case 0b01000: return (DECODED_INSTR(AMOOR_D));
          case 0b01100: return (DECODED_INSTR(AMOAND_D));
          case 0b10000: return (DECODED_INSTR(AMOMIN_D));
          case 0b10100: return (DECODED_INSTR(AMOMAX_D));
          case 0b11000: return (DECODED_INSTR(AMOMINU_D));
          case 0b11100: return (DECODED_INSTR(AMOMAXU_D));
          }
          break;
        }
      }
#endif
    }
#ifdef RISCV_EXT_COMPRESSED
  } else if constexpr (compressed_enabled) {
    // RISC-V Compressed Extension
    const rv32c_instruction ci{instruction};
    switch (ci.opcode()) {
      // Quadrant 0
    case CI_CODE(0b000, 0b00):
      // if all bits are zero, it's an illegal instruction
      if (ci.whole != 0x0) {
        return (DECODED_INSTR(C0_ADDI4SPN));
      }
      return (DECODED_INSTR(ILLEGAL));
    case CI_CODE(0b001, 0b00):
    case CI_CODE(0b010, 0b00):
    case CI_CODE(0b011, 0b00):
      if (ci.CL.funct3 == 0x1) { // C.FLD
        return (DECODED_INSTR(C0_REG_FLD));
      } else if (ci.CL.funct3 == 0x2) { // C.LW
        return (DECODED_INSTR(C0_REG_LW));
      } else if (ci.CL.funct3 == 0x3) {
        if constexpr (sizeof(address_t) == 8) { // C.LD
          return (DECODED_INSTR(C0_REG_LD));
        } else { // C.FLW
          return (DECODED_INSTR(C0_REG_FLW));
        }
      }
      return (DECODED_INSTR(ILLEGAL));
    // RESERVED: 0b100, 0b00
    case CI_CODE(0b101, 0b00):
    case CI_CODE(0b110, 0b00):
    case CI_CODE(0b111, 0b00):
      switch (ci.CS.funct3) {
      case 4: return (DECODED_INSTR(UNIMPLEMENTED));
      case 5: // C.FSD
        return (DECODED_INSTR(C0_REG_FSD));
      case 6: // C.SW
        return (DECODED_INSTR(C0_REG_SW));
      case 7: // C.SD / C.FSW
        if constexpr (sizeof(address_t) == 8) {
          return (DECODED_INSTR(C0_REG_SD));
        } else {
          return (DECODED_INSTR(C0_REG_FSW));
        }
      }
      return (DECODED_INSTR(ILLEGAL));
    // Quadrant 1
    case CI_CODE(0b000, 0b01): // C.ADDI
      if (ci.CI.rd != 0) {
        return (DECODED_INSTR(C1_ADDI));
      }
      return (DECODED_INSTR(NOP));
    case CI_CODE(0b001, 0b01): // C.ADDIW / C.JAL
      if constexpr (sizeof(address_t) == 8) {
        if (ci.CI.rd != 0) {
          return (DECODED_INSTR(C1_ADDIW));
        }
        return (DECODED_INSTR(NOP));
      } else {
        return (DECODED_INSTR(C1_JAL));
      }
    case CI_CODE(0b010, 0b01):
      if (ci.CI.rd != 0) {
        return (DECODED_INSTR(C1_LI));
      }
      return (DECODED_INSTR(NOP));
    case CI_CODE(0b011, 0b01):
      if (ci.CI.rd == 2) {
        return (DECODED_INSTR(C1_ADDI16SP));
      } else if (ci.CI.rd != 0) {
        return (DECODED_INSTR(C1_LUI));
      }
      return (DECODED_INSTR(ILLEGAL));
    case CI_CODE(0b100, 0b01): return (DECODED_INSTR(C1_ALU_OPS));
    case CI_CODE(0b101, 0b01): return (DECODED_INSTR(C1_JUMP));
    case CI_CODE(0b110, 0b01): return (DECODED_INSTR(C1_BEQZ));
    case CI_CODE(0b111, 0b01): return (DECODED_INSTR(C1_BNEZ));
    // Quadrant 2
    case CI_CODE(0b000, 0b10):
    case CI_CODE(0b001, 0b10):
    case CI_CODE(0b010, 0b10):
    case CI_CODE(0b011, 0b10):
      if (ci.CI.funct3 == 0x0 && ci.CI.rd != 0) {
        // C.SLLI
        return (DECODED_INSTR(C2_SLLI));
      } else if (ci.CI2.funct3 == 0x1) {
        // C.FLDSP
        return (DECODED_INSTR(C2_FLDSP));
      } else if (ci.CI2.funct3 == 0x2 && ci.CI2.rd != 0) {
        // C.LWSP
        return (DECODED_INSTR(C2_LWSP));
      } else if (ci.CI2.funct3 == 0x3) {
        if constexpr (sizeof(address_t) == 8) {
          if (ci.CI2.rd != 0) {
            // C.LDSP
            return (DECODED_INSTR(C2_LDSP));
          }
        } else {
          // C.FLWSP
          return (DECODED_INSTR(C2_FLWSP));
        }
      } else if (ci.CI.rd == 0) {
        // C.HINT
        return (DECODED_INSTR(NOP));
      }
      return (DECODED_INSTR(UNIMPLEMENTED));
    case CI_CODE(0b100, 0b10): {
      const bool topbit = ci.whole & (1 << 12);
      if (!topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0) { // JR rd
        return (DECODED_INSTR(C2_JR));
      } else if (topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0) { // JALR ra, rd+0
        return (DECODED_INSTR(C2_JALR));
      } else if (!topbit && ci.CR.rd != 0 && ci.CR.rs2 != 0) { // MV rd, rs2
        return (DECODED_INSTR(C2_MV));
      } else if (ci.CR.rd != 0) { // ADD rd, rd + rs2
        return (DECODED_INSTR(C2_ADD));
      } else if (topbit && ci.CR.rd == 0 && ci.CR.rs2 == 0) { // EBREAK
        return (DECODED_INSTR(C2_EBREAK));
      }
      return (DECODED_INSTR(UNIMPLEMENTED));
    }
    case CI_CODE(0b101, 0b10):
    case CI_CODE(0b110, 0b10):
    case CI_CODE(0b111, 0b10):
      if (ci.CSS.funct3 == 5) {
        // FSDSP
        return (DECODED_INSTR(C2_FSDSP));
      } else if (ci.CSS.funct3 == 6) {
        // SWSP
        return (DECODED_INSTR(C2_SWSP));
      } else if (ci.CSS.funct3 == 7) {
        if constexpr (sizeof(address_t) == 8) {
          // SDSP
          return (DECODED_INSTR(C2_SDSP));
        } else {
          // FSWSP
          return (DECODED_INSTR(C2_FSWSP));
        }
      }
      return (DECODED_INSTR(UNIMPLEMENTED));
    }
  }
#endif
  // all zeroes: illegal instruction
  if (instruction.whole == 0x0) {
    return (DECODED_INSTR(ILLEGAL));
  }

  if (CPU<W>::on_unimplemented_instruction != nullptr) {
    return (CPU<W>::on_unimplemented_instruction(instruction));
  } else {
    return (DECODED_INSTR(UNIMPLEMENTED));
  }
}
namespace riscv
{
	template <> RISCV_INTERNAL
	const CPU<4>::instruction_t& CPU<4>::decode(const format_t instruction)
	{
    return fuck_this<4>(instruction);
  }

  template <> RISCV_INTERNAL void CPU<4>::execute(const format_t instruction) {
    auto dec = decode(instruction);
    dec.handler(*this, instruction);
  }

  template <> RISCV_INTERNAL void CPU<4>::execute(uint8_t &handler_idx, uint32_t instr) {
    if (handler_idx == 0 && instr != 0) {
      [[unlikely]];
      handler_idx = DecoderData<4>::handler_index_for(decode(instr).handler);
    }
    DecoderData<4>::get_handlers()[handler_idx](*this, instr);
  }

  template <> const Instruction<4> &CPU<4>::get_unimplemented_instruction() noexcept {
    return DECODED_INSTR(UNIMPLEMENTED);
  }

  template <> RISCV_COLD_PATH() std::string Registers<4>::to_string() const {
    char buffer[600];
    int len = 0;
    for (int i = 1; i < 32; i++) {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%s\t%08X] ", RISCV::regname(i), this->get(i));
      if (i % 5 == 4) {
        len += snprintf(buffer + len, sizeof(buffer) - len, "\n");
      }
    }
    return std::string(buffer, len);
  }

  template <>
  RISCV_COLD_PATH()
  std::string CPU<4>::to_string(instruction_format format, const instruction_t &instr) const {
    char buffer[256];
    char ibuffer[128];
    int ibuflen = instr.printer(ibuffer, sizeof(ibuffer), *this, format);
    int len = 0;
    if (format.length() == 4) {
      len = snprintf(buffer, sizeof(buffer), "[%08X] %08X %.*s", this->pc(), format.whole, ibuflen, ibuffer);
    } else if (format.length() == 2) {
      len = snprintf(buffer, sizeof(buffer), "[%08X]     %04hX %.*s", this->pc(), (uint16_t)format.whole, ibuflen,
                     ibuffer);
    } else {
      throw MachineException(UNIMPLEMENTED_INSTRUCTION_LENGTH, "Unimplemented instruction format length",
                             format.length());
    }
    return std::string(buffer, len);
  }
}
