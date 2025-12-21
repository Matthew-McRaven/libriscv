#pragma once

// #define SYSCALL_VERBOSE 1
#ifdef SYSCALL_VERBOSE
#define SYSPRINT(fmt, ...)                                                                                             \
  {                                                                                                                    \
    char syspbuf[1024];                                                                                                \
    machine.print(syspbuf, snprintf(syspbuf, sizeof(syspbuf), fmt, ##__VA_ARGS__));                                    \
  }
static constexpr bool verbose_syscalls = true;
#else
#define SYSPRINT(fmt, ...) /* fmt */
static constexpr bool verbose_syscalls = false;
#endif

#include <fcntl.h>
#include <libriscv/machine_impl.hpp>
#include <libriscv/memory/memory_inline.hpp>
#include <signal.h>
#undef sa_handler
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#if !defined(__OpenBSD__) && !defined(TARGET_OS_IPHONE)
#include <sys/random.h>
#endif
extern "C" int dup3(int oldfd, int newfd, int flags);
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/uio.h>
#if __has_include(<termios.h>)
#include <termios.h>
#endif
#include <sys/syscall.h>
#ifndef EBADFD
#define EBADFD EBADF // OpenBSD, FreeBSD
#endif
#define LINUX_SA_ONSTACK 0x08000000
