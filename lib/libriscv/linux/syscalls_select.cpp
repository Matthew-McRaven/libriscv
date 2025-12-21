#include <sys/select.h>

template <AddressType address_t>
static void syscall_pselect(Machine<address_t>&)
{
    throw MachineException(SYSTEM_CALL_FAILED, "pselect() not implemented");
}
