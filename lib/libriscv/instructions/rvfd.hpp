#pragma once
#include "rv32i_instr.hpp"

namespace riscv
{
	union rv32f_instruction
	{
		struct {
			uint32_t opcode : 7;
			uint32_t rd     : 5;
			uint32_t funct3 : 3;
			uint32_t rs1    : 5;
			uint32_t rs2    : 5;
			uint32_t funct7 : 7;
		} Rtype;
		struct {
			uint32_t opcode : 7;
			uint32_t rd     : 5;
			uint32_t funct3 : 3;
			uint32_t rs1    : 5;
			uint32_t rs2    : 5;
			uint32_t funct2 : 2;
			uint32_t rs3    : 5;
		} R4type;
		struct {
			uint32_t opcode : 7;
			uint32_t rd     : 5;
			uint32_t funct3 : 3;
			uint32_t rs1    : 5;
			uint32_t imm    : 12;

			bool sign() const noexcept {
				return imm & 0x800;
			}
			int32_t signed_imm() const noexcept {
				return int32_t(imm << 20) >> 20;
			}
		} Itype;
		struct {
			uint32_t opcode : 7;
			uint32_t imm04  : 5;
			uint32_t funct3 : 3;
			uint32_t rs1    : 5;
			uint32_t rs2    : 5;
			uint32_t imm510 : 6;
			uint32_t imm11  : 1;

			bool sign() const noexcept {
				return imm11;
			}
			int32_t signed_imm() const noexcept {
				const int32_t imm = imm04 | (imm510 << 5) | (imm11 << 11);
				return (imm << 20) >> 20;
			}
		} Stype;

		uint16_t half[2];
		uint32_t whole;

		rv32f_instruction(rv32i_instruction i) : whole(i.whole) {}

		uint32_t opcode() const noexcept {
			return Rtype.opcode;
		}
	};
	static_assert(sizeof(rv32f_instruction) == 4, "Must be 4 bytes");

	enum fflags {
		FFLAG_NX = 0x1,
		FFLAG_UF = 0x2,
		FFLAG_OF = 0x4,
		FFLAG_DZ = 0x8,
		FFLAG_NV = 0x10
	};
}

#ifdef RISCV_32I
extern const riscv::Instruction<4> instr32i_FLW;
extern const riscv::Instruction<4> instr32i_FLD;
extern const riscv::Instruction<4> instr32i_FSW;
extern const riscv::Instruction<4> instr32i_FSD;
extern const riscv::Instruction<4> instr32i_FMADD;
extern const riscv::Instruction<4> instr32i_FMSUB;
extern const riscv::Instruction<4> instr32i_FNMADD;
extern const riscv::Instruction<4> instr32i_FNMSUB;
extern const riscv::Instruction<4> instr32i_FADD;
extern const riscv::Instruction<4> instr32i_FSUB;
extern const riscv::Instruction<4> instr32i_FMUL;
extern const riscv::Instruction<4> instr32i_FDIV;
extern const riscv::Instruction<4> instr32i_FSQRT;
extern const riscv::Instruction<4> instr32i_FMIN_FMAX;
extern const riscv::Instruction<4> instr32i_FEQ_FLT_FLE;
extern const riscv::Instruction<4> instr32i_FCVT_SD_DS;
extern const riscv::Instruction<4> instr32i_FCVT_W_SD;
extern const riscv::Instruction<4> instr32i_FCVT_SD_W;
extern const riscv::Instruction<4> instr32i_FSGNJ_NX;
extern const riscv::Instruction<4> instr32i_FCLASS;
extern const riscv::Instruction<4> instr32i_FMV_X_W;
extern const riscv::Instruction<4> instr32i_FMV_W_X;

extern const riscv::Instruction<8> instr64i_FLW;
extern const riscv::Instruction<8> instr64i_FLD;
extern const riscv::Instruction<8> instr64i_FSW;
extern const riscv::Instruction<8> instr64i_FSD;
extern const riscv::Instruction<8> instr64i_FMADD;
extern const riscv::Instruction<8> instr64i_FMSUB;
extern const riscv::Instruction<8> instr64i_FNMADD;
extern const riscv::Instruction<8> instr64i_FNMSUB;
extern const riscv::Instruction<8> instr64i_FADD;
extern const riscv::Instruction<8> instr64i_FSUB;
extern const riscv::Instruction<8> instr64i_FMUL;
extern const riscv::Instruction<8> instr64i_FDIV;
extern const riscv::Instruction<8> instr64i_FSQRT;
extern const riscv::Instruction<8> instr64i_FMIN_FMAX;
extern const riscv::Instruction<8> instr64i_FEQ_FLT_FLE;
extern const riscv::Instruction<8> instr64i_FCVT_SD_DS;
extern const riscv::Instruction<8> instr64i_FCVT_SD_W;
extern const riscv::Instruction<8> instr64i_FSGNJ_NX;
extern const riscv::Instruction<8> instr64i_FCLASS;
extern const riscv::Instruction<8> instr64i_FMV_X_W;
extern const riscv::Instruction<8> instr64i_FMV_W_X;
#endif
