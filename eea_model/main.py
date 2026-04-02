from register import Registers
from one_iter import one_iter
from one_iter_opt import one_iter_opt

import pandas as pd
import random
import math

# from ecdsa import SECP256k1

if __name__ == "__main__":

    # --------------------------------------------------------------
    # Configuration section
    # Select the modulus p and the target integer x.
    # You may use a random x or import a large curve prime (e.g., SECP256k1).
    # And choose the iteration function to use:
    #   one_iter     – baseline version (fully explicit phase transitions)
    #   one_iter_opt – optimized version (reduce quantum gate complexity)
    # --------------------------------------------------------------

    p = 10000000343
    # p = SECP256k1.curve.p()
    x = 413568799
    # x = random.randint(1, p - 1)

    # FUNC = one_iter
    FUNC = one_iter_opt

    # --------------------------------------------------------------
    # Main algorithm
    # --------------------------------------------------------------

    x_ = x
    iter_ = 0
    if x > (p >> 1):
        x_ = p - x
        iter_ = 1
    
    registers = Registers(p, x_, iter_)
    columns = [
        "work1", "work2", "|",
        "t", "q", "r", "|",
        "t_prime", "r_prime", "|",
        "l_t", "l_q", "l_r_prime", "l_shift", "|",
        "phase1", "phase2", "iter", "sign"
    ]
    df = pd.DataFrame(columns=columns)

    # According to our complexity bound, the number of iterations is approximately 4 / log_2(phi), where phi is the golden ratio.
    const = 1 / math.log((math.sqrt(5) + 1) / 2, 2)
    num_iters = 4 * math.ceil(const * registers.n)

    df.loc[len(df)] = registers.snapshot()
    # For the quantum version, x must satisfy x <= p / 2. Otherwise, we use p - x and flip the iteration parity qubit.
    # This ensures the algorithm remains reversible.
    if x > (p >> 1):
        registers.work2.r_prime = p - x
        registers.control.iter = 1
    df.loc[len(df)] = registers.snapshot()

    # Main iteration loop
    for i in range(num_iters):
        FUNC(registers)
        df.loc[len(df)] = registers.snapshot()

    # Result verification and output
    # The modular inverse is obtained from Work2.t_prime, adjusted by the iteration parity bit.
    print("-" * 40)
    x_inverse = registers.work2.t_prime * (2 * (registers.control.iter % 2) - 1) % p
    check = x * x_inverse % p

    print(f"Result:")
    print(f"  p          = {p}")
    print(f"  x          = {x}")
    print(f"  x_inverse  = {x_inverse}")
    print(f"x * x_inverse mod p = {check}")
    print("-" * 40)

    df.to_csv("result.csv")