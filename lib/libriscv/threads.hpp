#pragma once
#include <cstdio>
#include <unordered_map>
#include <vector>
#include "./core/registers.hpp"
#include "./types.hpp"

namespace riscv {

template <AddressType> struct Machine;

template <AddressType> struct MultiThreading;
static const uint32_t PARENT_SETTID  = 0x00100000; /* set the TID in the parent */
static const uint32_t CHILD_CLEARTID = 0x00200000; /* clear the TID in the child */
static const uint32_t CHILD_SETTID   = 0x01000000; /* set the TID in the child */

//#define THREADS_DEBUG 1
#ifdef THREADS_DEBUG
#define THPRINT(machine, fmt, ...) \
	{ char thrpbuf[1024]; machine.print(thrpbuf, \
		snprintf(thrpbuf, sizeof(thrpbuf), fmt, ##__VA_ARGS__)); }
#else
#define THPRINT(fmt, ...) /* fmt */
#endif

template <AddressType address_t> struct Thread {

  MultiThreading<address_t> &threading;
  const int tid;
  // For returning to this thread
  Registers<address_t> stored_regs;
  // Base address of the stack
  address_t stack_base;
  // Size of the stack
  address_t stack_size;
  // Address zeroed when exiting
  address_t clear_tid = 0;
  // The current or last blocked word
  uint32_t block_word = 0;
  uint32_t block_extra = 0;

  Thread(MultiThreading<address_t> &, int tid, address_t tls, address_t stack, address_t stkbase, address_t stksize);
  Thread(MultiThreading<address_t> &, const Thread &other);
  bool exit(); // Returns false when we *cannot* continue
  void suspend();
  void suspend(address_t return_value);
  void block(uint32_t reason, uint32_t extra = 0);
  void block_return(address_t return_value, uint32_t reason, uint32_t extra);
	void activate();
	void resume();
};

template <AddressType address_t> struct MultiThreading {
  using thread_t = Thread<address_t>;

  thread_t *create(int flags, address_t ctid, address_t ptid, address_t stack, address_t tls, address_t stkbase,
                   address_t stksize);
  int get_tid() const noexcept { return m_current->tid; }
  thread_t *get_thread();
  thread_t *get_thread(int tid); /* or nullptr */
  bool preempt();
  bool      suspend_and_yield(long result = 0);
	bool      yield_to(int tid, bool store_retval = true);
	void      erase_thread(int tid);
	void      wakeup_next();
	bool      block(address_t retval, uint32_t reason, uint32_t extra = 0);
	void      unblock(int tid);
	size_t    wakeup_blocked(size_t max, uint32_t reason, uint32_t mask = ~0U);
	/* A suspended thread can at any time be resumed. */
	auto&     suspended_threads() { return m_suspended; }
	/* A blocked thread can only be resumed by unblocking it. */
	auto&     blocked_threads() { return m_blocked; }

  MultiThreading(Machine<address_t> &);
  MultiThreading(Machine<address_t> &, const MultiThreading &);
  Machine<address_t> &machine;
  std::vector<thread_t*> m_blocked;
	std::vector<thread_t*> m_suspended;
	std::unordered_map<int, thread_t> m_threads;
	unsigned   m_thread_counter = 0;
	unsigned   m_max_threads = 50;
	thread_t*  m_current = nullptr;
};
} // riscv
