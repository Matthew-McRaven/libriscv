#pragma once

template <int W>
template <typename T> inline
T Memory<W>::read(address_t address)
{
	return *(T*) main_memory().ro_at(address, sizeof(T));
}

template <int W>
template <typename T> inline
T& Memory<W>::writable_read(address_t address)
{
	return *(T*) main_memory().rw_at(address, sizeof(T));
}

template <int W>
template <typename T> inline
void Memory<W>::write(address_t address, T value)
{
	*(T*) main_memory().rw_at(address, sizeof(T)) = value;
}

template <int W>
inline address_type<W> Memory<W>::resolve_address(const std::string& name) const
{
	auto* sym = resolve_symbol(name.c_str());
	return (sym) ? sym->st_value : 0x0;
}

template <int W>
inline address_type<W> Memory<W>::resolve_section(const char* name) const
{
	auto* shdr = this->section_by_name(name);
	if (shdr) return shdr->sh_addr;
	return 0x0;
}

template <int W>
inline address_type<W> Memory<W>::exit_address() const noexcept
{
	return this->m_exit_address;
}

template <int W>
inline void Memory<W>::set_exit_address(address_t addr)
{
	this->m_exit_address = addr;
}
