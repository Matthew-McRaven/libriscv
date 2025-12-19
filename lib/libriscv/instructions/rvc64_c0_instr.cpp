#include "rvc_helpers.hpp"

const riscv::Instruction<8> instr64i_C0_ADDI4SPN{riscv::C0_ADDI4SPN_handler, riscv::C0_ADDI4SPN_printer};
const riscv::Instruction<8> instr64i_C0_REG_FLD{riscv::C0_REG_FLD_handler, riscv::C0_REG_FLD_printer};
const riscv::Instruction<8> instr64i_C0_REG_LW{riscv::C0_REG_LW_handler, riscv::C0_REG_FLD_printer};
const riscv::Instruction<8> instr64i_C0_REG_LD{riscv::C0_REG_LD_handler, riscv::C0_REG_FLD_printer};
const riscv::Instruction<8> instr64i_C0_REG_FLW{riscv::C0_REG_FLW_handler, riscv::C0_REG_FLD_printer};
const riscv::Instruction<8> instr64i_C0_REG_FSD{riscv::C0_REG_FSD_handler, riscv::C0_REG_FSD_printer};
const riscv::Instruction<8> instr64i_C0_REG_SW{riscv::C0_REG_SW_handler, riscv::C0_REG_FSD_printer};
const riscv::Instruction<8> instr64i_C0_REG_SD{riscv::C0_REG_SD_handler, riscv::C0_REG_FSD_printer};
const riscv::Instruction<8> instr64i_C0_REG_FSW{riscv::C0_REG_FSW_handler, riscv::C0_REG_FSD_printer};
