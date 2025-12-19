#include "decoder_cache.hpp"
// #define ENABLE_TIMINGS

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

bool SegmentKey::operator==(const SegmentKey &other) const { return pc == other.pc && crc == other.crc; }

bool SegmentKey::operator<(const SegmentKey &other) const {
  return pc < other.pc || (pc == other.pc && crc < other.crc);
}

} // namespace riscv
