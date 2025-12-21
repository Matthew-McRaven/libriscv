#include "common.hpp"
#include "core/decode/decoder_cache.hpp"

#include "core/decode/decoder_cache_impl.hpp"
#include "machine_impl.hpp"

namespace riscv {

INSTANTIATE_32_IF_ENABLED(DecoderData);
INSTANTIATE_64_IF_ENABLED(DecoderData);
} // namespace riscv
