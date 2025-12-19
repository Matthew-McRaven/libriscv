#include "decoder_cache.hpp"
#include <inttypes.h>
#include <mutex>
#include <unordered_set>
#include "instructions/instruction_list.hpp"
#include "instructions/rvc.hpp"
#include "instructions/safe_instr_loader.hpp"
#include "internal_common.hpp"
#include "machine.hpp"
#include "threaded_bytecodes.hpp"
#include "util/crc32.hpp"
//#define ENABLE_TIMINGS

namespace riscv
{

#ifdef ENABLE_TIMINGS
timespec time_now() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t;
}
long nanodiff(timespec start_time, timespec end_time) {
  return (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
}
#endif

INSTANTIATE_32_IF_ENABLED(DecoderData);
INSTANTIATE_32_IF_ENABLED(Memory);
INSTANTIATE_32_IF_ENABLED(CPU);
INSTANTIATE_64_IF_ENABLED(DecoderData);
INSTANTIATE_64_IF_ENABLED(Memory);
INSTANTIATE_64_IF_ENABLED(CPU);
INSTANTIATE_128_IF_ENABLED(DecoderData);
INSTANTIATE_128_IF_ENABLED(Memory);
INSTANTIATE_128_IF_ENABLED(CPU);

bool SegmentKey::operator==(const SegmentKey &other) const { return pc == other.pc && crc == other.crc; }

bool SegmentKey::operator<(const SegmentKey &other) const {
  return pc < other.pc || (pc == other.pc && crc < other.crc);
}

} // namespace riscv
