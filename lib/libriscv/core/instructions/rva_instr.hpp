#pragma once

#include <cstdint>
#include "../../types.hpp"

extern const riscv::Instruction<uint32_t> instr32i_AMOADD_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOXOR_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOOR_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOAND_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOMAX_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOMIN_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOMAXU_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOMINU_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOADD_D;
extern const riscv::Instruction<uint32_t> instr32i_AMOXOR_D;
extern const riscv::Instruction<uint32_t> instr32i_AMOOR_D;
extern const riscv::Instruction<uint32_t> instr32i_AMOAND_D;
extern const riscv::Instruction<uint32_t> instr32i_AMOMAX_D;
extern const riscv::Instruction<uint32_t> instr32i_AMOMIN_D;
extern const riscv::Instruction<uint32_t> instr32i_AMOMAXU_D;
extern const riscv::Instruction<uint32_t> instr32i_AMOMINU_D;
extern const riscv::Instruction<uint32_t> instr32i_AMOSWAP_W;
extern const riscv::Instruction<uint32_t> instr32i_AMOSWAP_D;
extern const riscv::Instruction<uint32_t> instr32i_LOAD_RESV;
extern const riscv::Instruction<uint32_t> instr32i_STORE_COND;

extern const riscv::Instruction<uint64_t> instr64i_AMOADD_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOXOR_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOOR_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOAND_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOMAX_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOMIN_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOMAXU_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOMINU_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOADD_D;
extern const riscv::Instruction<uint64_t> instr64i_AMOXOR_D;
extern const riscv::Instruction<uint64_t> instr64i_AMOOR_D;
extern const riscv::Instruction<uint64_t> instr64i_AMOAND_D;
extern const riscv::Instruction<uint64_t> instr64i_AMOMAX_D;
extern const riscv::Instruction<uint64_t> instr64i_AMOMIN_D;
extern const riscv::Instruction<uint64_t> instr64i_AMOMAXU_D;
extern const riscv::Instruction<uint64_t> instr64i_AMOMINU_D;
extern const riscv::Instruction<uint64_t> instr64i_AMOSWAP_W;
extern const riscv::Instruction<uint64_t> instr64i_AMOSWAP_D;
extern const riscv::Instruction<uint64_t> instr64i_LOAD_RESV;
extern const riscv::Instruction<uint64_t> instr64i_STORE_COND;
