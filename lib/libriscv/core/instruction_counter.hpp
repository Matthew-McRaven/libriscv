#include <cstdint>

namespace riscv
{
    template <AddressType address_t> struct Machine;

	// In fastsim mode the instruction counter becomes a register
	// the function, and we only update m_counter in Machine on exit
	// When binary translation is enabled we cannot do this optimization.
	struct InstrCounter
	{
		InstrCounter(uint64_t icounter, uint64_t maxcounter)
		  : m_counter(icounter),
			m_max(maxcounter)
		{}
		~InstrCounter() = default;

		template <AddressType address_t>
		void apply(Machine<address_t>& machine) {
			machine.set_instruction_counter(m_counter);
			machine.set_max_instructions(m_max);
		}
		template <AddressType address_t>
		void apply_counter(Machine<address_t>& machine) {
			machine.set_instruction_counter(m_counter);
		}
		// Used by binary translator to compensate for its own function already being counted
		// TODO: Account for this inside the binary translator instead. Very minor impact.
		template <AddressType address_t>
		void apply_counter_minus_1(Machine<address_t>& machine) {
			machine.set_instruction_counter(m_counter-1);
			machine.set_max_instructions(m_max);
		}
		template <AddressType address_t>
		void retrieve_max_counter(Machine<address_t>& machine) {
			m_max     = machine.max_instructions();
		}
		template <AddressType address_t>
		void retrieve_counters(Machine<address_t>& machine) {
			m_counter = machine.instruction_counter();
			m_max     = machine.max_instructions();
		}

		uint64_t value() const noexcept {
			return m_counter;
		}
		uint64_t max() const noexcept {
			return m_max;
		}
		void stop() noexcept {
			m_max = 0; // This stops the machine
		}
		void set_counters(uint64_t value, uint64_t max) {
			m_counter = value;
			m_max     = max;
		}
		void increment_counter(uint64_t cnt) {
			m_counter += cnt;
		}
		bool overflowed() const noexcept {
			return m_counter >= m_max;
		}
	private:
		uint64_t m_counter;
		uint64_t m_max;
	};
} // riscv
