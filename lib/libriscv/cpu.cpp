#include "decoder_cache.hpp"
#include "instruction_counter.hpp"
#include "instructions/rv32i_instr.hpp"
#include "internal_common.hpp"
#include "machine.hpp"
#include "riscvbase.hpp"
#include "threaded_bytecodes.hpp"
//#define TIME_EXECUTION

namespace riscv
{
#ifdef TIME_EXECUTION
	static timespec time_now()
	{
		timespec t;
		clock_gettime(CLOCK_MONOTONIC, &t);
		return t;
	}
	static long nanodiff(timespec start_time, timespec end_time)
	{
		return (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
	}
#endif
	INSTANTIATE_32_IF_ENABLED(CPU);
	INSTANTIATE_32_IF_ENABLED(Registers);
	INSTANTIATE_64_IF_ENABLED(CPU);
	INSTANTIATE_64_IF_ENABLED(Registers);
	INSTANTIATE_128_IF_ENABLED(CPU);
	INSTANTIATE_128_IF_ENABLED(Registers);
}
