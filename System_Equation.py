from sympy import symbols, Function, sin
from modulpythonus.sym.eq.pde import PDE

class MassSpringDamperPDE(PDE):
    def __init__(self):
        t = symbols('t')
        x = Function('x')(t)
        m, c, k = 1.0, 0.5, 2.0  # Use your system's parameters
        self.equations = {
            "msd": m * x.diff(t, 2) + c * x.diff(t) + k * x - sin(2*t)
        }
        super().__init__(self.equations)