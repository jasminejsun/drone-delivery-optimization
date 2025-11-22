import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# node coordinates
node_coords = {
    0: (35, 35),  # depot
    1: (40.356, 36.287),
    2: (29.768, 33.453),
    3: (36.925, 27.514),
}

N = len(node_coords)
node_ids = list(node_coords.keys())

# cost coefficients
c1 = 10     # UAV startup cost
c2 = 0.3    # distance cost coefficient
c3 = 0.66   # energy cost coefficient

# energy model parameters
alpha = 101.18
beta  = -1381.73
wu = 36      # UAV weight (kg)

# generate distance matrix d[i,j]
d = np.zeros((N, N))
for i in node_ids:
    x_i, y_i = node_coords[i]
    for j in node_ids:
        x_j, y_j = node_coords[j]
        d[i,j] = np.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)

# Number of decision variables (x[i,j] continuous relax)
num_vars = N * N

def unpack_x(x_vec):
    """Convert flattened vector x into (N,N) matrix."""
    return x_vec.reshape((N, N))


def energy_ij(q_ij):
    """Energy model f_ij = alpha*(wu + q_ij) + beta"""
    return alpha * (wu + q_ij) + beta

TAS = 14.0  # m/s, from paper

def objective(x_vec):
    X = unpack_x(x_vec)

    # startup
    F1 = c1 * np.sum(X[0, :])

    # distance
    F2 = c2 * np.sum(X * d)

    q = 2.0  # example payload per leg (kg)
    f_val = energy_ij(q)  # same for all arcs in this simple version

    # d is in km -> energy(kWh) per arc:
    # E_ij = f_val * d_ij / (TAS * 3600)
    # (derivation: time = d_km*1000 / TAS, then divide by 3.6e6 to get kWh)
    E_matrix = f_val * d / (TAS * 3600.0)

    # energy
    F3 = c3 * np.sum(E_matrix * X)

    return F1 + F2 + F3

bounds = [(0, 1)] * num_vars

# forces the UAV to leave the depot exactly once
def depot_flow_constraint(x_vec):
    X = unpack_x(x_vec)
    return np.sum(X[0, :]) - 1.0   # = 0

constraints = [
    {"type": "eq", "fun": depot_flow_constraint},
]

# initial guess
x0 = np.ones(num_vars) * 0.1

solution = minimize(
    objective,
    x0,
    bounds=bounds,
    constraints=constraints,
)

print("Success:", solution.success)
print("Total cost:", solution.fun)
print("x matrix:\n", unpack_x(solution.x))

