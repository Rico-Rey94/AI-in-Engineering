import torch
import torch.nn as nn
from Mass_Spring_Damper_Data import generate_msd_data

X, Y = generate_msd_data()

class MSDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = MSDNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_t = torch.tensor(X, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)

for epoch in range(3000):
    optimizer.zero_grad()
    Y_pred = model(X_t)
    loss = loss_fn(Y_pred, Y_t)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.6f}")

with torch.no_grad():
    test_sample = X_t[:5]
    pred = model(test_sample)
    print("Predicted:\n", pred)
    print("True:\n", Y_t[:5])
