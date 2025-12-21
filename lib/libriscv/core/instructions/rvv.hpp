#pragma once
#include "rv32i_instr.hpp"

namespace riscv
{
	union rv32v_instruction
	{
		// Vector Load
		struct {
			uint32_t opcode : 7;
			uint32_t vd     : 5;
			uint32_t width  : 3;
			uint32_t rs1    : 5;
			uint32_t lumop  : 5;
			uint32_t vm     : 1;
			uint32_t mop    : 2;
			uint32_t mew    : 1;
			uint32_t nf     : 3;
		} VL;
		struct {
			uint32_t opcode : 7;
			uint32_t vd     : 5;
			uint32_t width  : 3;
			uint32_t rs1    : 5;
			uint32_t rs2    : 5;
			uint32_t vm     : 1;
			uint32_t mop    : 2;
			uint32_t mew    : 1;
			uint32_t nf     : 3;
		} VLS;
		struct {
			uint32_t opcode : 7;
			uint32_t vd     : 5;
			uint32_t width  : 3;
			uint32_t rs1    : 5;
			uint32_t vs2    : 5;
			uint32_t vm     : 1;
			uint32_t mop    : 2;
			uint32_t mew    : 1;
			uint32_t nf     : 3;
		} VLX;

		// Vector Store
		struct {
			uint32_t opcode : 7;
			uint32_t vs3    : 5;
			uint32_t width  : 3;
			uint32_t rs1    : 5;
			uint32_t sumop  : 5;
			uint32_t vm     : 1;
			uint32_t mop    : 2;
			uint32_t mew    : 1;
			uint32_t nf     : 3;
		} VS;
		struct {
			uint32_t opcode : 7;
			uint32_t vs3    : 5;
			uint32_t width  : 3;
			uint32_t rs1    : 5;
			uint32_t rs2    : 5;
			uint32_t vm     : 1;
			uint32_t mop    : 2;
			uint32_t mew    : 1;
			uint32_t nf     : 3;
		} VSS;
		struct {
			uint32_t opcode : 7;
			uint32_t vs3    : 5;
			uint32_t width  : 3;
			uint32_t rs1    : 5;
			uint32_t vs2    : 5;
			uint32_t vm     : 1;
			uint32_t mop    : 2;
			uint32_t mew    : 1;
			uint32_t nf     : 3;
		} VSX;

		// Vector Configuration
		struct {
			uint32_t opcode : 7;
			uint32_t rd     : 5;
			uint32_t ones   : 3;
			uint32_t rs1    : 5;
			uint32_t zimm   : 12;
		} VLI; // 0, 1 && 0, 0
		struct {
			uint32_t opcode : 7;
			uint32_t rd     : 5;
			uint32_t ones   : 3;
			uint32_t uimm   : 5;
			uint32_t zimm   : 12;
		} IVLI; // 1, 1
		struct {
			uint32_t opcode : 7;
			uint32_t rd     : 5;
			uint32_t ones   : 3;
			uint32_t rs1    : 5;
			uint32_t rs2    : 5;
			uint32_t zimm   : 7;
		} VSETVL; // 1, 0

		struct {
			uint32_t opcode : 7;
			uint32_t vd     : 5;
			uint32_t funct3 : 3;
			uint32_t vs1    : 5;
			uint32_t vs2    : 5;
			uint32_t vm     : 1;
			uint32_t funct6 : 6;
		} OPVV;

		struct {
			uint32_t opcode : 7;
			uint32_t vd     : 5;
			uint32_t funct3 : 3;
			uint32_t imm    : 5;
			uint32_t vs2    : 5;
			uint32_t vm     : 1;
			uint32_t funct6 : 6;
		} OPVI;

		rv32v_instruction(const rv32i_instruction& i) : whole(i.whole) {}
		uint32_t whole;
	};
	static_assert(sizeof(rv32v_instruction) == 4, "Instructions are 32-bits");

} // riscv

#ifdef RISCV_32I
extern const riscv::Instruction<uint32_t> instr32i_VSETVLI;
extern const riscv::Instruction<uint32_t> instr32i_VSETIVLI;
extern const riscv::Instruction<uint32_t> instr32i_VSETVL;
extern const riscv::Instruction<uint32_t> instr32i_VLE32;
extern const riscv::Instruction<uint32_t> instr32i_VSE32;
extern const riscv::Instruction<uint32_t> instr32i_VOPI_VV;
extern const riscv::Instruction<uint32_t> instr32i_VOPF_VV;
extern const riscv::Instruction<uint32_t> instr32i_VOPM_VV;
extern const riscv::Instruction<uint32_t> instr32i_VOPI_VI;
extern const riscv::Instruction<uint32_t> instr32i_VOPF_VF;

extern const riscv::Instruction<uint64_t> instr64i_VSETVLI;
extern const riscv::Instruction<uint64_t> instr64i_VSETIVLI;
extern const riscv::Instruction<uint64_t> instr64i_VSETVL;
extern const riscv::Instruction<uint64_t> instr64i_VLE32;
extern const riscv::Instruction<uint64_t> instr64i_VSE32;
extern const riscv::Instruction<uint64_t> instr64i_VOPI_VV;
extern const riscv::Instruction<uint64_t> instr64i_VOPF_VV;
extern const riscv::Instruction<uint64_t> instr64i_VOPM_VV;
extern const riscv::Instruction<uint64_t> instr64i_VOPI_VI;
extern const riscv::Instruction<uint64_t> instr64i_VOPF_VF;
#endif
