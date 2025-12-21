#pragma once
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>
#include "libriscv_settings.h"

namespace riscv
{

template <class T>
concept AddressType = std::same_as<T, std::uint32_t> || std::same_as<T, std::uint64_t>;
template <class T>
concept SignedAddressType = std::same_as<T, std::uint32_t> || std::same_as<T, std::uint64_t>;
template <class T>
concept AnyAddressType = AddressType<T> || SignedAddressType<T>;
// unsigned -> signed (same width)
template <AddressType T>
using ToSignedAddress = std::conditional_t<std::same_as<T, std::uint32_t>, std::int32_t, std::int64_t>;

// signed -> unsigned (same width)
template <SignedAddressType T>
using ToAddress = std::conditional_t<std::same_as<T, std::int32_t>, std::uint32_t, std::uint64_t>;

template <AnyAddressType T>
using flip_signedness_t = std::conditional_t<AddressType<T>, ToSignedAddress<T>, ToAddress<T>>;

template <AddressType> struct CPU;

enum exceptions {
  ILLEGAL_OPCODE,
  ILLEGAL_OPERATION,
  PROTECTION_FAULT,
  EXECUTION_SPACE_PROTECTION_FAULT,
  MISALIGNED_INSTRUCTION,
  UNIMPLEMENTED_INSTRUCTION_LENGTH,
  UNIMPLEMENTED_INSTRUCTION,
  UNHANDLED_SYSCALL,
  OUT_OF_MEMORY,
  INVALID_ALIGNMENT,
  DEADLOCK_REACHED,
  MAX_INSTRUCTIONS_REACHED,
  FEATURE_DISABLED,
  INVALID_PROGRAM,
  SYSTEM_CALL_FAILED,
  EXECUTION_LOOP_DETECTED,
  UNKNOWN_EXCEPTION
};

using instruction_format = union rv32i_instruction;
template <AddressType address_type> using instruction_handler = void (*)(CPU<address_type> &, instruction_format);
template <AddressType address_type>
using instruction_printer = int (*)(char *, size_t, const CPU<address_type> &, instruction_format);
template <AddressType address_type> using register_type = address_type;

template <AddressType address_type> struct Instruction {
  instruction_handler<address_type> handler; // callback for executing one instruction
  instruction_printer<address_type> printer; // callback for logging one instruction
};

class MachineException : public std::exception {
public:
  explicit MachineException(const int type, const char *text, const uint64_t data = 0)
      : m_type{type}, m_data{data}, m_msg{text} {}

  virtual ~MachineException() throw() {}

  int type() const throw() { return m_type; }
  uint64_t data() const throw() { return m_data; }
  const char *what() const throw() override { return m_msg; }

protected:
  const int m_type;
  const uint64_t m_data;
  const char *m_msg;
};

class MachineTimeoutException : public MachineException {
  using MachineException::MachineException;
};

enum trapmode {
  TRAP_READ = 0x0,
  TRAP_WRITE = 0x1000,
  TRAP_EXEC = 0x2000,
};

template <AddressType> struct DecoderData;

// Instructions may be unaligned with C-extension
// On amd64 we take the cost, because it's faster
union UnderAlign32 {
  uint16_t data[2];
  operator uint32_t() { return data[0] | uint32_t(data[1]) << 16; }
};
}
