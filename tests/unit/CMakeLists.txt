cmake_minimum_required(VERSION 3.10)
project(libriscv_unittests CXX)

set(SAN_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")

# Verify we can compile+link a tiny program with both compile and link options.
set(CMAKE_REQUIRED_OPTIONS "${SAN_FLAGS}")
set(CMAKE_REQUIRED_LINK_OPTIONS "${SAN_FLAGS}")
check_cxx_source_compiles("int main(){return 0;}" HAS_ASAN_UBSAN)

if(HAS_ASAN_UBSAN)
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -O1 -ggdb3 ${SAN_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -O1 -ggdb3")
endif()

option(RISCV_MULTIPROCESS "" ON)

target_compile_definitions(
  riscv PUBLIC FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION=1
               RISCV_SYSCALLS_MAX=600)

enable_testing()

function(add_unit_test NAME)
  add_executable(${NAME} ${ARGN} codebuilder.cpp)
  target_compile_definitions(${NAME}
                             PUBLIC SRCDIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${NAME} riscv catch catch-main)
  add_test(NAME test_${NAME} COMMAND ${NAME})
endfunction()

add_unit_test(basic basic.cpp)
add_unit_test(basic32 basic32.cpp)
add_unit_test(brutal brutal.cpp)
add_unit_test(checksum checksum.cpp)
add_unit_test(crc32c)
add_unit_test(custom custom.cpp)
add_unit_test(dynamic dynamic.cpp)
add_unit_test(examples examples.cpp)
add_unit_test(heap heaptest.cpp)
add_unit_test(fptest fp_testsuite.cpp)
add_unit_test(micro micro.cpp)
add_unit_test(memtrap memory_trap.cpp)
add_unit_test(native native.cpp)
# add_unit_test(png png.cpp)
add_unit_test(protect protections.cpp)
add_unit_test(rvbuffer rvbuffer.cpp)
add_unit_test(serialize serialize.cpp)
add_unit_test(vmcall vmcall.cpp)
add_unit_test(va_exec va_execute.cpp)
add_unit_test(elftest verify_elf.cpp)
