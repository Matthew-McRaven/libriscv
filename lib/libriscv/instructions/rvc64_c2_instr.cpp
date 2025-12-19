#include "rvc_helpers.hpp"

const riscv::Instruction<8> instr64i_C2_SLLI{riscv::C2_SLLI_handler, riscv::C2_SLLI_printer};
const riscv::Instruction<8> instr64i_C2_FLDSP{riscv::C2_FLDSP_handler, riscv::C2_SLLI_printer};
const riscv::Instruction<8> instr64i_C2_LWSP{riscv::C2_LWSP_handler, riscv::C2_SLLI_printer};
const riscv::Instruction<8> instr64i_C2_LDSP{riscv::C2_LDSP_handler, riscv::C2_SLLI_printer};
const riscv::Instruction<8> instr64i_C2_FLWSP{riscv::C2_FLWSP_handler, riscv::C2_SLLI_printer};
const riscv::Instruction<8> instr64i_C2_FSDSP{riscv::C2_FSDSP_handler, riscv::C2_FSDSP_printer};
const riscv::Instruction<8> instr64i_C2_SWSP{riscv::C2_SWSP_handler, riscv::C2_FSDSP_printer};
const riscv::Instruction<8> instr64i_C2_SDSP{riscv::C2_SDSP_handler, riscv::C2_SDSP_printer};
const riscv::Instruction<8> instr64i_C2_FSWSP{riscv::C2_FSWSP_handler, riscv::C2_FSDSP_printer};
const riscv::Instruction<8> instr64i_C2_JR{riscv::C2_JR_handler, riscv::C2_JR_printer};
const riscv::Instruction<8> instr64i_C2_JALR{riscv::C2_JALR_handler, riscv::C2_JR_printer};
const riscv::Instruction<8> instr64i_C2_MV{riscv::C2_MV_handler, riscv::C2_JR_printer};
const riscv::Instruction<8> instr64i_C2_ADD{riscv::C2_ADD_handler, riscv::C2_JR_printer};
const riscv::Instruction<8> instr64i_C2_EBREAK{riscv::C2_EBREAK_handler, riscv::C2_JR_printer};
