from register import length, Registers

"""
One iteration of the space-efficient Extended Euclidean Algorithm (EEA) under classical simulation.
-----------------------------------------------------------------------
This implementation provides a step-by-step simulation of one EEA iteration, explicitly separating
the four algorithmic phases. Each operation is designed to be quantum-implementable; that is,
all state transitions arereversible and correspond to unitary transformations in the quantum setting.

During the transition from Phase 3 to Phase 4, the control bit 'sign' is explicitly set to 1 to
preserve reversibility of the switching operation. This adjustment ensures that the overall process
remains unitary, consistent with the quantum constraint that all operations must be invertible.
"""
def one_iter(registers: Registers):
    w1, w2, ctrl = registers.work1, registers.work2, registers.control

    # --------------------------------------------------------------
    # Arithmetic logic implementing the four phases of the algorithm
    # --------------------------------------------------------------
    if not ctrl.phase1 and not ctrl.phase2:
        """Quantumly shift register Work2 left by one bit position"""
        w2.l_s = w2.l_s + 1
        # Comparison (by subtraction and addition) acting on qubits [l_t + l_q + 2, n + 3 - l_s] of Work1 and Work2.
        ctrl.sign = (ctrl.sign + (w1.r < (w2.r_prime << w2.l_s))) % 2

    if not ctrl.phase1 and ctrl.phase2:
        """Quantumly shift register Work2 right by one bit position"""
        w2.l_s = w2.l_s - 1
        w1.l_q = w1.l_q + 1
        # Subtraction acting on qubits [l_t + l_q + 2, n + 3 - l_s] of Work1 and Work2, with sign bit activated.
        w1.r = w1.r - (w2.r_prime << w2.l_s)
        ctrl.sign = (ctrl.sign + (w1.r < 0)) % 2
        # Conditional addition acting on the qubits [l_t + l_q + 2, n + 3 - l_s] of Work1 and Work2.
        if ctrl.sign:
            w1.r = w1.r + (w2.r_prime << w2.l_s)
        ctrl.sign = (ctrl.sign + 1) % 2
        # Controlled SWAP operation acting on the (l_t + l_q + 1)-th bit of Work1 and Sign qubit.
        if ctrl.sign:
            w1.q = w1.q + (1 << w2.l_s); ctrl.sign = 0

    if ctrl.phase1 and not ctrl.phase2:
        # Controlled SWAP operation acting on the (l_t + l_q + 1)-th bit of Work1 and Sign qubit.
        if (w1.q >> w2.l_s) % 2:
            ctrl.sign = 1; w1.q = w1.q - (1 << w2.l_s)
        # Conditional subtraction acting on the first (l_t + 1) qubits of Work1 and Work2.
        if not ctrl.sign:
            w2.t_prime = w2.t_prime - (w1.t << w2.l_s)
        ctrl.sign = (ctrl.sign + 1) % 2
        # Addition acting on the first (l_t + 1) qubits of Work1 and Work2, with sign bit activated.
        ctrl.sign = (ctrl.sign + (w2.t_prime < 0)) % 2
        w2.t_prime = w2.t_prime + (w1.t << w2.l_s)
        """Quantumly shift register Work2 left by one bit position"""
        w2.l_s = w2.l_s + 1
        w1.l_q = w1.l_q - 1

    if ctrl.phase1 and ctrl.phase2:
        # Comparison (by subtraction and addition) acting on the first (l_t + 1) qubits of Work1 and Work2.
        ctrl.sign = (ctrl.sign + (w2.t_prime >= (w1.t << w2.l_s))) % 2
        """Quantumly shift register Work2 right by one bit position"""
        w2.l_s = w2.l_s - 1
    
    # --------------------------------------------------------------
    # Phase update logic (state transition between the four phases)
    # --------------------------------------------------------------
    if not w1.l_q and w2.l_r_prime:
        ctrl.phase2 = (ctrl.phase2 + ctrl.sign + ctrl.phase1) % 2
        ctrl.sign = (ctrl.sign + ctrl.phase2) % 2
    if not w2.l_s:
        ctrl.phase1 = (ctrl.phase1 + 1) % 2
        ctrl.phase2 = (ctrl.phase2 + 1) % 2
    
    # --------------------------------------------------------------
    # Register swaping at the end of one EEA iteration
    # --------------------------------------------------------------
    if not w1.l_q and not w2.l_s:
        # Performs an n-qubit SWAP between Work1 and Work2.
        tmp = w1.t; w1.t = w2.t_prime; w2.t_prime = tmp
        tmp = w1.r; w1.r = w2.r_prime; w2.r_prime = tmp
        
        # Updates length indicators after SWAP.
        w1.l_t = w1.l_t - length(w2.t_prime)
        w1.l_t = w1.l_t + length(w1.t)
        w2.l_r_prime = w2.l_r_prime - length(w1.r)
        w2.l_r_prime = w2.l_r_prime + length(w2.r_prime)

        # Flip the iteration parity qubit.
        ctrl.iter = (ctrl.iter + 1) % 2