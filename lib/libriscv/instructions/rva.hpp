#pragma once

#include <cstdint>
#include "../common.hpp"
#include "../types.hpp"

#define AMOSIZE_W 0x2
#define AMOSIZE_D 0x3
#define AMOSIZE_Q 0x4

namespace riscv
{
	template <int W>
	struct AtomicMemory
	{
		using address_t = address_type<W>;

    bool load_reserve(int size, address_t addr) RISCV_INTERNAL {
      if (!check_alignment(size, addr)) return false;

      m_reservation = addr;
			return true;
    }

    // Volume I: RISC-V Unprivileged ISA V20190608 p.49:
    // An SC can only pair with the most recent LR in program order.
    bool store_conditional(int size, address_t addr) RISCV_INTERNAL
		{
			if (!check_alignment(size, addr))
				return false;

			bool result = m_reservation == addr;
			// Regardless of success or failure, executing an SC.W
			// instruction invalidates any reservation held by this hart.
			m_reservation = 0x0;
			return result;
		}

	private:
		inline bool check_alignment(int size, address_t addr) RISCV_INTERNAL
		{
			return (addr & (size-1)) == 0;
		}

		address_t m_reservation = 0x0;
	};
}

#ifdef RISCV_32I
extern const riscv::Instruction<4> instr32i_AMOADD_W;
extern const riscv::Instruction<4> instr32i_AMOXOR_W;
extern const riscv::Instruction<4> instr32i_AMOOR_W;
extern const riscv::Instruction<4> instr32i_AMOAND_W;
extern const riscv::Instruction<4> instr32i_AMOMAX_W;
extern const riscv::Instruction<4> instr32i_AMOMIN_W;
extern const riscv::Instruction<4> instr32i_AMOMAXU_W;
extern const riscv::Instruction<4> instr32i_AMOMINU_W;
extern const riscv::Instruction<4> instr32i_AMOADD_D;
extern const riscv::Instruction<4> instr32i_AMOXOR_D;
extern const riscv::Instruction<4> instr32i_AMOOR_D;
extern const riscv::Instruction<4> instr32i_AMOAND_D;
extern const riscv::Instruction<4> instr32i_AMOMAX_D;
extern const riscv::Instruction<4> instr32i_AMOMIN_D;
extern const riscv::Instruction<4> instr32i_AMOMAXU_D;
extern const riscv::Instruction<4> instr32i_AMOMINU_D;
extern const riscv::Instruction<4> instr32i_AMOSWAP_W;
extern const riscv::Instruction<4> instr32i_AMOSWAP_D;
extern const riscv::Instruction<4> instr32i_LOAD_RESV;
extern const riscv::Instruction<4> instr32i_STORE_COND;

extern const riscv::Instruction<8> instr64i_AMOADD_W;
extern const riscv::Instruction<8> instr64i_AMOXOR_W;
extern const riscv::Instruction<8> instr64i_AMOOR_W;
extern const riscv::Instruction<8> instr64i_AMOAND_W;
extern const riscv::Instruction<8> instr64i_AMOMAX_W;
extern const riscv::Instruction<8> instr64i_AMOMIN_W;
extern const riscv::Instruction<8> instr64i_AMOMAXU_W;
extern const riscv::Instruction<8> instr64i_AMOMINU_W;
extern const riscv::Instruction<8> instr64i_AMOADD_D;
extern const riscv::Instruction<8> instr64i_AMOXOR_D;
extern const riscv::Instruction<8> instr64i_AMOOR_D;
extern const riscv::Instruction<8> instr64i_AMOAND_D;
extern const riscv::Instruction<8> instr64i_AMOMAX_D;
extern const riscv::Instruction<8> instr64i_AMOMIN_D;
extern const riscv::Instruction<8> instr64i_AMOMAXU_D;
extern const riscv::Instruction<8> instr64i_AMOMINU_D;
extern const riscv::Instruction<8> instr64i_AMOSWAP_W;
extern const riscv::Instruction<8> instr64i_AMOSWAP_D;
extern const riscv::Instruction<8> instr64i_LOAD_RESV;
extern const riscv::Instruction<8> instr64i_STORE_COND;
#endif
