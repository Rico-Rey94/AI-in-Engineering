import numpy as np
from scipy.integrate import solve_ivp

m = 1.0     # kg
c = 0.5     # Ns/m
k = 2.0     # N/m

def force(t):
    return np.sin(2*t)

def msd(t, state):
    x, v = state
    dxdt = v
    dvdt = (force(t) - c*v - k*x) / m
    return [dxdt, dvdt]

t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)
x0 = [0.0, 0.0]

sol = solve_ivp(msd, t_span, x0, t_eval=t_eval)

x = sol.y[0]
v = sol.y[1]
u = force(t_eval)

X = np.column_stack([x, v, u])
Y = np.column_stack([v, (u - c*v - k*x)/m])  # true derivatives
