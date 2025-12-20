#include "decoder_cache.hpp"
#include "internal_common.hpp"
#include "memory.hpp"

#include "decoder_cache_impl.hpp"
#include "machine_impl.hpp"
#include "memory_inline.hpp"

namespace riscv {
INSTANTIATE_32_IF_ENABLED(Memory);
INSTANTIATE_64_IF_ENABLED(Memory);

INSTANTIATE_32_IF_ENABLED(DecoderData);
INSTANTIATE_64_IF_ENABLED(DecoderData);
} // namespace riscv
