#pragma once

template <int W>
void Memory<W>::memset(address_t dst, uint8_t value, size_t len)
{
	__builtin_memset(m_main_memory.rw_at(dst, len), value, len);
}

template <int W>
void Memory<W>::memcpy(address_t dst, const void* src, size_t len)
{
	__builtin_memcpy(m_main_memory.rw_at(dst, len), src, len);
}
template <int W>
void Memory<W>::memcpy(address_t dst, Machine<W>& srcm, address_t src, address_t len)
{
	const char* srcp = srcm.memory.m_main_memory.ro_at(src, len);
	this->memcpy(dst, srcp, len);
}

template <int W>
void Memory<W>::memcpy_out(void* dst, address_t src, size_t len) const
{
	__builtin_memcpy(dst, m_main_memory.ro_at(src, len), len);
}

template <int W>
std::string_view Memory<W>::memview(address_t addr, size_t len) const
{
	return {m_main_memory.ro_at(addr, len), len};
}

template <int W>
std::string Memory<W>::memstring(address_t addr, const size_t max_len) const
{
	const size_t len = strlen(addr, max_len);
	return std::string {m_main_memory.ro_at(addr, len), len};
}

template <int W>
size_t Memory<W>::strlen(address_t addr, size_t max_len) const
{
	max_len = std::min(m_main_memory.max_length(addr), max_len);
	const char* src = m_main_memory.ro_at(addr, max_len);
	return strnlen(src, max_len);
}

template <int W>
int Memory<W>::memcmp(address_t a1, address_t a2, size_t len) const
{
	const char* m1 = m_main_memory.ro_at(a1, len);
	const char* m2 = m_main_memory.ro_at(a2, len);
	return __builtin_memcmp(m1, m2, len);
}
template <int W>
int Memory<W>::memcmp(const void* src, address_t a2, size_t len) const
{
	const char* m2 = m_main_memory.ro_at(a2, len);
	return __builtin_memcmp(src, m2, len);
}
