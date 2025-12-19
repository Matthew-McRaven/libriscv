#include "cpu.hpp"
#include "decoder_cache.hpp"
#include "internal_common.hpp"
#include "memory.hpp"

namespace riscv {
INSTANTIATE_32_IF_ENABLED(CPU);
INSTANTIATE_64_IF_ENABLED(CPU);

INSTANTIATE_32_IF_ENABLED(Memory);
INSTANTIATE_64_IF_ENABLED(Memory);

INSTANTIATE_32_IF_ENABLED(DecoderData);
INSTANTIATE_64_IF_ENABLED(DecoderData);
} // namespace riscv
