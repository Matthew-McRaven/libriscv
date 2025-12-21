#include "common.hpp"
#include "decode/decoder_cache.hpp"
#include "memory.hpp"

#include "decode/decoder_cache_impl.hpp"
#include "machine_impl.hpp"
#include "memory_inline.hpp"

namespace riscv {
INSTANTIATE_32_IF_ENABLED(Memory);
INSTANTIATE_64_IF_ENABLED(Memory);

INSTANTIATE_32_IF_ENABLED(DecoderData);
INSTANTIATE_64_IF_ENABLED(DecoderData);
} // namespace riscv
