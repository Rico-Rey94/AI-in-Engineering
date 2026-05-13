import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# parameters
m, c, k = 10**6, 10**6, 10**8

# generate reference data
t_eval = np.linspace(0, 10, 10000)

def force_np(t):
    return np.sin(2*t)

def msd(t, state):
    x, v = state
    return [v, (force_np(t) - c*v - k*x)/m]

sol = solve_ivp(msd, (0,10), [0,0], t_eval=t_eval)

t_data = torch.tensor(t_eval[:, None], dtype=torch.float32)
x_true = torch.tensor(sol.y[0][:, None], dtype=torch.float32)

# model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        return self.net(t)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# losses
def physics_loss(model, t):
    t.requires_grad_(True)
    x = model(t)
    dx = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    ddx = torch.autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
    F = torch.sin(2*t)
    residual = m*ddx + c*dx + k*x - F
    return torch.mean(residual**2)

def data_loss(model):
    return torch.mean((model(t_data) - x_true)**2)


# training
loss_history = []
for epoch in range(50000):
    optimizer.zero_grad()
    loss = physics_loss(model, t_data) + 10*data_loss(model)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.6f}")

# Plotting Loss vs Epoch
plt.figure(figsize=(8, 6))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Mass-Spring-Damper - Loss vs. Epoch (Epoch: {epoch})")
plt.yscale("log")  # <- This will use a logarithmic scale for the loss
plt.grid(True)
plt.show()
