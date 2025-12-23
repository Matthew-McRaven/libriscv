#include <sys/select.h>
#include "../../common.hpp"
#include "../../machine.hpp"

namespace riscv {
template <AddressType address_t>
static void syscall_pselect(Machine<address_t>&)
{
    throw MachineException(SYSTEM_CALL_FAILED, "pselect() not implemented");
}
} // namespace riscv
