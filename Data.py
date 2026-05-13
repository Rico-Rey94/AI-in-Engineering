import numpy as np
from Mass_Spring_Damper_Data import generate_msd_data

X, Y = generate_msd_data()  # outputs (N, 3), (N, 2)
# Assume t_eval is available or returned from generate_msd_data

t_eval = np.linspace(0, 10, 1000)

# Create dicts
data_dict = {
    "t": t_eval.reshape(-1, 1),  # shape (N, 1)
    "x": X[:, 0:1],              # shape (N, 1)
    "v": X[:, 1:2],              # shape (N, 1)
    "u": X[:, 2:3],              # shape (N, 1)
}
deriv_dict = {
    "dx/dt": Y[:, 0:1],
    "dv/dt": Y[:, 1:2],
}