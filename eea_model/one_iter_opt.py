from register import length, Registers

"""
Optimized single-iteration implementation of the quantum-friendly Extended Euclidean Algorithm (EEA)
under classical simulation.
-----------------------------------------------------------------------
In the baseline version, several redundant operations increase the overall quantum gate complexity,
particularly the shifts and the duplicated comparison steps:
    - Between Phases 1 and 2:
        'ctrl.sign = (ctrl.sign + (w1.r < (w2.r_prime << w2.l_s))) % 2'
    - Between Phases 3 and 4:
        'ctrl.sign = (ctrl.sign + (w2.t_prime >= (w1.t << w2.l_s))) % 2'
This optimized version removes such redundancies to improve circuit efficiency.

Another key optimization merges the two location-controlled SWAP operations (in Phases 2 and 3)
into a single operation. While a location-controlled SWAP gate is relatively expensive 
(O(n log n) gates in a naive design, or O(n log log n) in an optimized implementation),
the shifts are comparatively cheap (O(n) gates). By strategically introducing additional
shift operations, the total cost is minimized.

Overall asymptotic cost:
    O(n log log n):
        • 1 location-controlled SWAP
        • 2 location-controlled comparisons
        • 1 location-controlled addition
        • 1 location-controlled subtraction
        • 4 length-update operations
    O(n):
        • 4 n-qubit shifts
        • 1 n-qubit SWAP
"""
def one_iter_opt(registers: Registers):
    w1, w2, ctrl = registers.work1, registers.work2, registers.control

    # --------------------------------------------------------------
    # Pre-shift operations
    # --------------------------------------------------------------
    if not ctrl.phase1:
        """Quantumly shift register Work2 left by one bit position"""
        w2.l_s = w2.l_s + 1
    if not ctrl.phase1 and ctrl.phase2:
        """Quantumly shift register Work2 right by two bit positions"""
        w2.l_s = w2.l_s - 2
    
    # --------------------------------------------------------------
    # Arithmetic block 1: location-controlled subtraction on r's
    # --------------------------------------------------------------
    if not ctrl.phase1:
        # Subtraction acting on qubits [l_t + l_q + 2, n + 3 - l_s] of Work1 and Work2, with sign bit activated.
        w1.r = w1.r - (w2.r_prime << w2.l_s)
        ctrl.sign = (ctrl.sign + (w1.r < 0)) % 2
    if not ctrl.phase1 and ctrl.phase2:
        ctrl.sign = (ctrl.sign + 1) % 2
    if not ctrl.phase1 and (not ctrl.phase2 or not ctrl.sign):
        # Conditional addition acting on qubits [l_t + l_q + 2, n + 3 - l_s] of Work1 and Work2.
        w1.r = w1.r + (w2.r_prime << w2.l_s)
    
    # --------------------------------------------------------------
    # Arithmetic block 2: location-controlled SWAP between 'sign' and the corresponding qubit of quotient
    # --------------------------------------------------------------
    ctrl.phase2 = (ctrl.phase2 + ctrl.phase1) % 2
    if ctrl.phase2:
        # Controlled SWAP operation acting on the (l_t + l_q)-th bit of Work1 and Sign qubit.
        if ctrl.sign and not (w1.q >> w2.l_s) % 2:
            ctrl.sign = 0; w1.q = w1.q + (1 << w2.l_s)
        elif not ctrl.sign and (w1.q >> w2.l_s) % 2:
            ctrl.sign = 1; w1.q = w1.q - (1 << w2.l_s)

        if ctrl.phase1: w1.l_q = w1.l_q - 1
        else: w1.l_q = w1.l_q + 1
    ctrl.phase2 = (ctrl.phase2 + ctrl.phase1) % 2

    # --------------------------------------------------------------
    # Arithmetic block 3: location-controlled addition on t's
    # --------------------------------------------------------------
    if ctrl.phase1 and (ctrl.phase2 or not ctrl.sign):
        # Conditional subtraction acting on the first (l_t + 1) qubits of Work1 and Work2.
        w2.t_prime = w2.t_prime - (w1.t << w2.l_s)
    if ctrl.phase1:
        ctrl.sign = (ctrl.sign + 1) % 2
        # Addition acting on the first (l_t + 1) qubits of Work1 and Work2, with sign bit activated.
        ctrl.sign = (ctrl.sign + (w2.t_prime < 0)) % 2
        w2.t_prime = w2.t_prime + (w1.t << w2.l_s)
    
    # --------------------------------------------------------------
    # Post-shift operations
    # --------------------------------------------------------------
    if ctrl.phase1:
        """Shift Work2 left by 1"""
        w2.l_s = w2.l_s + 1
    if ctrl.phase1 and ctrl.phase2:
        """Shift Work2 right by 2"""
        w2.l_s = w2.l_s - 2
    
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