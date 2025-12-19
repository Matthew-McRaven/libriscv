#include "machine.hpp"

#include "decoder_cache.hpp"
#include "internal_common.hpp"

namespace riscv {

// The zero-page and guarded page share backing
static const Page zeroed_page{PageAttributes{.read = true, .write = false, .exec = false, .is_cow = true}};
static const Page guarded_page{
    PageAttributes{.read = false, .write = false, .exec = false, .is_cow = false, .non_owning = true},
    zeroed_page.m_page.get()};
static const Page host_codepage{
    PageAttributes{.read = false, .write = false, .exec = true, .is_cow = false, .non_owning = true},
    std::array<uint8_t, PageSize>{// STOP: 0x7ff00073
                                  0x73, 0x00, 0xf0, 0x7f,
                                  // JMP -4 (jump back to STOP): 0xffdff06f
                                  0x6f, 0xf0, 0xdf, 0xff, 0x0}};
const Page &Page::cow_page() noexcept {
  return zeroed_page; // read-only, zeroed page
}
const Page &Page::guard_page() noexcept {
  return guarded_page; // inaccessible page
}
const Page &Page::host_page() noexcept {
  return host_codepage; // host code page
}

INSTANTIATE_32_IF_ENABLED(Memory);
INSTANTIATE_64_IF_ENABLED(Memory);
INSTANTIATE_128_IF_ENABLED(Memory);
} // namespace riscv
