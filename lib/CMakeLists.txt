#
# C++20 RISC-V emulator library
#

# DEBUG allows memory alignment checks and other things
option(RISCV_DEBUG "Enable extra checks in the RISC-V machine" OFF)
# Enable and disable various RISC-V instruction set extensions. Not recommended
# to disable any.
option(RISCV_EXT_A "Enable RISC-V atomic instructions" ON)
option(RISCV_EXT_C "Enable RISC-V compressed instructions" ON)
option(RISCV_EXT_V "Enable RISC-V vector instructions" ON)
# Enable 32- and 64-bit architecture emulation
option(RISCV_32I "Enable 32-bit RISC-V" ON)
option(RISCV_64I "Enable 64-bit RISC-V" ON)

# Enable Floating-point Control and Status Register emulation
option(RISCV_FCSR "Enable FCSR emulation" OFF)
# EXPERIMENTAL enables some high-performance interpreter features that may be
# unstable.
option(RISCV_EXPERIMENTAL "Enable experimental features" OFF)
# MEMORY_TRAPS allows you to trap writes to uncacheable pages in memory. Cached
# pages can only be trapped once.
option(RISCV_MEMORY_TRAPS "Enable memory page traps" ON)
# FLAT_RW_ARENA simplifies the program arena, making the heap area always
# read-write. Memory before the heap and outside of the arena behaves like
# before.
option(RISCV_FLAT_RW_ARENA "Enable performant flat read-write arena" ON)

set(THREADED_IS_DEFAULT OFF)
# Threaded simulation uses computed goto, and is not supported on all
# compilers/systems.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}"
                                                  STREQUAL "GNU")
  option(RISCV_THREADED "Enable fastest simulation mode" ON)
else()
  option(RISCV_THREADED "Enable fastest simulation mode" OFF)
endif()

# Version information from git tags
set(RISCV_VERSION_MAJOR 1)
set(RISCV_VERSION_MINOR 11)

set(SOURCES
    libriscv/debug.cpp
    libriscv/decoder_cache.cpp
    libriscv/memory.cpp
    libriscv/explicit_templates.cpp
    libriscv/posix/minimal.cpp
    libriscv/posix/signals.cpp
    libriscv/posix/threads.cpp
    libriscv/posix/socket_calls.cpp
    libriscv/serialize.cpp
    libriscv/instructions/rvc_instr.cpp
    libriscv/instructions/rvi_instr.cpp
    libriscv/instructions/rvf_instr.cpp
    libriscv/instructions/rva_instr.cpp
    libriscv/instructions/rvv_instr.cpp
    libriscv/util/crc32c.cpp)

if(MINGW_TOOLCHAIN
   OR MINGW
   OR WIN32)
  list(APPEND SOURCES libriscv/win32/system_calls.cpp)
elseif(UNIX)
  list(APPEND SOURCES libriscv/linux/system_calls.cpp)
endif()

if(RISCV_THREADED)
  message(STATUS "libriscv: Threaded dispatch enabled")
  list(APPEND SOURCES libriscv/threaded_dispatch.cpp
       libriscv/threaded_inaccurate_dispatch.cpp)
else()
  message(STATUS "libriscv: Switch-based dispatch enabled")
  list(APPEND SOURCES libriscv/bytecode_dispatch.cpp)
endif()

# Generate the settings header RISCV_VERSION_MAJOR and RISCV_VERSION_MINOR are
# required by libriscv_settings.h.in
configure_file(libriscv_settings.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/libriscv_settings.h)
configure_file(libriscv.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libriscv.pc)

add_library(riscv ${SOURCES})
if(COSMOPOLITAN)
  target_compile_options(riscv PUBLIC -fexceptions)
endif()

if(APPLE)
  find_library(SECURITY_FRAMEWORK Security)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  mark_as_advanced(SECURITY_FRAMEWORK FOUNDATION_FRAMEWORK)

  target_link_libraries(riscv PUBLIC "${SECURITY_FRAMEWORK}"
                                     "${FOUNDATION_FRAMEWORK}")
endif()

target_compile_features(riscv PUBLIC cxx_std_20)
target_include_directories(riscv PUBLIC .)
target_include_directories(riscv PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

if(NOT WIN32 OR MINGW_TOOLCHAIN)
  target_compile_options(riscv PRIVATE -Wall -Wextra)
endif()

if(RISCV_EXPERIMENTAL AND RISCV_ENCOMPASSING_ARENA)
  target_compile_definitions(
    riscv PUBLIC RISCV_ENCOMPASSING_ARENA_BITS=${RISCV_ENCOMPASSING_ARENA_BITS})
endif()

if(RISCV_MULTIPROCESS)
  find_package(Threads REQUIRED)
  target_link_libraries(riscv PUBLIC Threads::Threads)
endif()

if(WIN32 OR MINGW_TOOLCHAIN)
  target_link_libraries(riscv PUBLIC wsock32 ws2_32)
endif()

if(${CMAKE_PROJECT_NAME} STREQUAL "libriscv")
  install(TARGETS riscv)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libriscv_settings.h
          DESTINATION include/${PROJECT_NAME})
  install(
    FILES libriscv/cached_address.hpp
          libriscv/common.hpp
          libriscv/cpu.hpp
          libriscv/cpu_inline.hpp
          libriscv/debug.hpp
          libriscv/decoded_exec_segment.hpp
          libriscv/decoder_cache.hpp
          libriscv/elf.hpp
          libriscv/guest_datatypes.hpp
          libriscv/instr_helpers.hpp
          libriscv/instruction_counter.hpp
          libriscv/instructions/instruction_list.hpp
          libriscv/machine.hpp
          libriscv/machine_inline.hpp
          libriscv/machine_vmcall.hpp
          libriscv/memory.hpp
          libriscv/memory_helpers_paging.hpp
          libriscv/memory_inline.hpp
          libriscv/memory_inline_pages.hpp
          libriscv/mmap_cache.hpp
          libriscv/native_heap.hpp
          libriscv/page.hpp
          libriscv/prepared_call.hpp
          libriscv/registers.hpp
          libriscv/rvv_registers.hpp
          libriscv/riscvbase.hpp
          libriscv/instructions/rv32i_instr.hpp
          libriscv/instructions/rva.hpp
          libriscv/instructions/rvc.hpp
          libriscv/instructions/rvfd.hpp
          libriscv/rsp_server.hpp
          libriscv/threads.hpp
          libriscv/types.hpp
    DESTINATION include/${PROJECT_NAME})
  install(FILES libriscv/linux/rsp_server.hpp
          DESTINATION include/${PROJECT_NAME}/linux)
  install(FILES libriscv/posix/filedesc.hpp libriscv/posix/signals.hpp
          DESTINATION include/${PROJECT_NAME}/posix)
  install(FILES libriscv/util/buffer.hpp libriscv/util/function.hpp
          DESTINATION include/${PROJECT_NAME}/util)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libriscv.pc
          DESTINATION lib/pkgconfig)
endif()
