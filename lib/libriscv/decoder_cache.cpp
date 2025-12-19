#include "machine.hpp"
#include "decoder_cache.hpp"
#include "instruction_list.hpp"
#include "internal_common.hpp"
#include "rvc.hpp"
#include "safe_instr_loader.hpp"
#include "threaded_bytecodes.hpp"
#include "util/crc32.hpp"
#include <inttypes.h>
#include <mutex>
#include <unordered_set>
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
