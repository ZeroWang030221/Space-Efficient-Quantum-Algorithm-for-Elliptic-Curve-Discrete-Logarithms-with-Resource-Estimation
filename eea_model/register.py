from typing import List

"""
Return the bit-length of an integer 'num'.
This function corresponds to updating the Length registers at the end of each EEA iteration,
since register allocation depends on the bit-width of current values.
"""
def length(num: int) -> int:
    if num == 0: return 0
    return len(bin(num)) - 2

class Work1:
    """
    Work1 quantum register model.
    Represents a (n+3)-qubit register storing the triple (t, q, r),
    where:
        - t: current t_i, stored in little-endian order (lower bits first);
        - q: intermediate quotient q' during phase 2 or 3, in big-endian order;
        - r: current r_{i-1}, or its updated form (r_{i-1} - q' * r_i), in big-endian order.
    """

    def __init__(self, n: int, t: int, q: int, r: int, l_t: int, l_q: int):
        self.n = n
        self.t = t
        self.q = q
        self.r = r
        self.l_t = l_t
        self.l_q = l_q

    def bin(self) -> str:
        """
        Return a bit-string visualization of Work1 register.
        The string layout is: [t (little-endian)] | [q (big-endian)] | [r (big-endian)].
        """
        str_t = str(bin(self.t))[:1:-1] if self.t != 0 else ""
        str_q = str(bin(self.q))[2:2 + self.l_q] if self.q != 0 else ""
        str_r = str(bin(self.r))[2:].zfill(self.n + 2 - self.l_t - self.l_q)
        return f"{str_t}0|{str_q}|{str_r}"

class Work2:
    """
    Work2 quantum register model.
    Represents another (n+3)-qubit register storing (t', r'),
    where:
        - t' = t_{i-1}, stored in little-endian order;
        - r' = r_i, stored in big-endian order.
    This register may undergo circular left shifts (by l_s positions) during phase 2 or 3
    to simulate bit reordering within superposed states.
    """

    def __init__(self, n: int, t_prime: int, r_prime: int, l_r_prime: int, l_s: int):
        self.n = n
        self.t_prime = t_prime
        self.r_prime = r_prime
        self.l_r_prime = l_r_prime
        self.l_s = l_s

    def bin(self) -> str:
        str_t_prime = str(bin(self.t_prime))[2:].zfill(self.n + 3 - self.l_r_prime)[::-1]
        str_r_prime = str(bin(self.r_prime))[2:] if self.r_prime != 0 else ""
        raw = f"{str_t_prime}|{str_r_prime}|"
        if self.l_r_prime == 0:
            return "Algorithm Terminated"
        return raw[self.l_s:] + raw[:self.l_s]

class Control:
    """
    Control register set.
    Contains several 1-qubit logical flags:
        - phase1, phase2: indicate which phase of the 4-phase algorithm is active;
        - iter: iteration parity flag of the EEA;
        - sign: used for sign control of intermediate results;
        - ctrl: general-purpose control.
    """
    def __init__(self, iter_: int, phase1: int = 0, phase2: int = 0, sign: int = 0):
        self.phase1 = phase1
        self.phase2 = phase2
        self.iter = iter_
        self.sign = sign


class Registers:
    def __init__(self, p: int, x: int, iter_: int):
        n = len(bin(p)) - 2
        logn = len(bin(n + 1)) - 2
        l_r_prime = len(bin(x)) - 2

        self.n = n
        self.logn = logn

        self.work1 = Work1(n, 1, 0, p, 1, 0)
        self.work2 = Work2(n, 0, x, l_r_prime, 0)
        self.control = Control(iter_)

    def snapshot(self) -> List:
        """Return a structured snapshot of current register states for visualization."""
        return [
            self.work1.bin(), self.work2.bin(), "|",
            self.work1.t, self.work1.q, self.work1.r, "|",
            self.work2.t_prime, self.work2.r_prime, "|", 
            self.work1.l_t, self.work1.l_q, self.work2.l_r_prime, self.work2.l_s, "|",
            self.control.phase1, self.control.phase2, self.control.iter, self.control.sign
        ]