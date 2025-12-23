#ifdef WIN32
#include "./win32/system_calls_impl.hpp"
#else
#include "./linux/system_calls_impl.hpp"
#endif
