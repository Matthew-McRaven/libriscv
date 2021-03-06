#include <string>

namespace riscv {
extern const std::string bintr_code =
R"123(#include <stdint.h>
#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#define ILLEGAL_OPCODE    0
#define PROTECTION_FAULT  2

#if RISCV_TRANSLATION_DYLIB == 4
	typedef uint32_t addr_t;
	typedef int32_t saddr_t;
#else
	typedef uint64_t addr_t;
	typedef int64_t saddr_t;
#endif

typedef union {
	int32_t i32[2];
	float   f32[2];
	int64_t i64;
	double  f64;
	struct {
		uint32_t bits  : 31;
		uint32_t sign  : 1;
		uint32_t upper;
	} lsign;
	struct {
		uint64_t bits  : 63;
		uint64_t sign  : 1;
	} usign;
} fp64reg;

static inline void load_fl(fp64reg* reg, uint32_t fv) {
	reg->i32[0] = fv;
	reg->i32[1] = 0;
}
static inline void set_fl(fp64reg* reg, float f) {
	reg->f32[0] = f;
	reg->i32[1] = 0;
}
#define load_dbl(reg, dv) (reg)->i64 = (dv)
#define set_dbl(reg, dv)  (reg)->f64 = (dv)

// Thin variant of CPU for higher compilation speed
__attribute__((aligned(16)))
typedef struct {
	addr_t  pc;
	addr_t  r[32];
	fp64reg fr[32];
} CPU;

static struct CallbackTable {
	uint32_t (*mem_ld32)(CPU*, addr_t);
	uint64_t (*mem_ld64)(CPU*, addr_t);
	void (*jump)(CPU*, addr_t, uint64_t);
	int  (*syscall)(CPU*, addr_t, uint64_t);
	void (*stop)(CPU*, uint64_t);
	void (*ebreak)(CPU*, uint64_t);
	void (*system)(CPU*, uint32_t);
	void (*exception)(CPU*, int, addr_t);
	float  (*sqrtf32)(float);
	double (*sqrtf64)(double);
} api;

static inline uint32_t SRA32(int is_signed, uint32_t shifts, uint32_t value)
{
	const uint32_t sign_bits = -is_signed ^ 0x0;
	const uint32_t sign_shifted = sign_bits << (32 - shifts);
	return (value >> shifts) | sign_shifted;
}
static inline uint64_t SRA64(int is_signed, uint32_t shifts, uint64_t value)
{
	const uint64_t sign_bits = -is_signed ^ 0x0;
	const uint64_t sign_shifted = sign_bits << (64 - shifts);
	return (value >> shifts) | sign_shifted;
}

// https://stackoverflow.com/questions/28868367/getting-the-high-part-of-64-bit-integer-multiplication
// As written by catid
static inline uint64_t MUL128(
	uint64_t* r_hi,
	const uint64_t x,
	const uint64_t y)
{
	const uint64_t x0 = (uint32_t)x, x1 = x >> 32;
	const uint64_t y0 = (uint32_t)y, y1 = y >> 32;
	const uint64_t p11 = x1 * y1, p01 = x0 * y1;
	const uint64_t p10 = x1 * y0, p00 = x0 * y0;

	// 64-bit product + two 32-bit values
	const uint64_t middle = p10 + (p00 >> 32) + (uint32_t)p01;

	// 64-bit product + two 32-bit values
	*r_hi = p11 + (middle >> 32) + (p01 >> 32);

	// Add LOW PART and lower half of MIDDLE PART
	return (middle << 32) | (uint32_t)p00;
}

static struct Memory {
	char *const mem;
	const addr_t physbase;
	const addr_t physend;
	const addr_t robase;
	const addr_t rwbase;
} *mem;

static inline int within(addr_t addr, addr_t asize, addr_t base) {
	return (addr >= base && addr + asize < mem->physend);
}
static inline uint8_t read8(CPU* cpu, addr_t addr) {
	if (UNLIKELY(!within(addr, 1, mem->robase)))
		api.exception(cpu, PROTECTION_FAULT, addr);
	return *(uint8_t *)&mem->mem[addr];
}
static inline uint16_t read16(CPU* cpu, addr_t addr) {
	if (UNLIKELY(!within(addr, 2, mem->robase)))
		api.exception(cpu, PROTECTION_FAULT, addr);
	return *(uint16_t *)&mem->mem[addr];
}
static inline uint32_t read32(CPU* cpu, addr_t addr) {
	if (UNLIKELY(!within(addr, 4, mem->robase)))
		api.exception(cpu, PROTECTION_FAULT, addr);
	return *(uint32_t *)&mem->mem[addr];
}
static inline uint64_t read64(CPU* cpu, addr_t addr) {
	if (UNLIKELY(!within(addr, 8, mem->robase)))
		api.exception(cpu, PROTECTION_FAULT, addr);
	return *(uint64_t *)&mem->mem[addr];
}

static char *getmem(CPU* cpu, addr_t addr, addr_t asize) {
	if (UNLIKELY(!within(addr, asize, mem->rwbase)))
		api.exception(cpu, PROTECTION_FAULT, addr);
	return &mem->mem[addr];
}
#define GETMEM(cpu, a, t) (t *)getmem(cpu, a, sizeof(t))

extern void init(struct CallbackTable* pcall, struct Memory* pmem) {
	api = *pcall;
	mem = pmem;
};
)123";
}
