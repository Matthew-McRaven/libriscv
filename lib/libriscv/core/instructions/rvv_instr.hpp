#pragma once
#include "../../types.hpp"

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
