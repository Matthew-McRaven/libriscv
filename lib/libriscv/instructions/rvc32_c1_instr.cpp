#include "rvc_helpers.hpp"

const riscv::Instruction<4> instr32i_C1_ADDI{riscv::C1_ADDI_handler, riscv::C1_ADDI_printer};
const riscv::Instruction<4> instr32i_C1_JAL{riscv::C1_JAL_handler, riscv::C1_JAL_printer};
const riscv::Instruction<4> instr32i_C1_ADDIW{riscv::C1_ADDIW_handler, riscv::C1_ADDIW_printer};
const riscv::Instruction<4> instr32i_C1_LI{riscv::C1_LI_handler, riscv::C1_LI_printer};
const riscv::Instruction<4> instr32i_C1_ADDI16SP{riscv::C1_ADDI16SP_handler, riscv::C1_ADDI16SP_printer};
const riscv::Instruction<4> instr32i_C1_LUI{riscv::C1_LUI_handler, riscv::C1_ADDI16SP_printer};
const riscv::Instruction<4> instr32i_C1_ALU_OPS{riscv::C1_ALU_OPS_handler, riscv::C1_ALU_OPS_printer};
const riscv::Instruction<4> instr32i_C1_JUMP{riscv::C1_JUMP_handler, riscv::C1_JUMP_printer};
const riscv::Instruction<4> instr32i_C1_BEQZ{riscv::C1_BEQZ_handler, riscv::C1_BEQZ_printer};
const riscv::Instruction<4> instr32i_C1_BNEZ{riscv::C1_BNEZ_handler, riscv::C1_BNEZ_printer};
