import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# data taken from simulation values from research paper

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

alpha = 101.18
beta  = -1381.73
wu = 36     

Q = 5.0    
Emax = 4.0  

# distance + energy

d = np.zeros((N, N))
for i in node_ids:
    x_i, y_i = node_coords[i]
    for j in node_ids:
        x_j, y_j = node_coords[j]
        d[i,j] = np.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)

def energy_ij(q_ij):
    return alpha * (wu + q_ij) + beta

TAS = 14.0  # m/s, from paper
q_leg = 2.0  
f_val = energy_ij(q_leg)

E_matrix = f_val * d / (TAS * 3600.0)

num_x = N * N
num_D = N
num_F = N
num_vars = num_x + num_D + num_F

def unpack_vars(z):
    X_flat = z[:num_x]
    D = z[num_x:num_x + num_D]
    F = z[num_x + num_D:]
    return X_flat.reshape((N, N)), D, F

# objective function

def objective(z):
    X, D, F = unpack_vars(z)
    F1 = c1 * np.sum(X[0, :])
    F2 = c2 * np.sum(X * d)
    F3 = c3 * np.sum(E_matrix * X)
    return F1 + F2 + F3

# path constraints

def depot_depart(z):
    X, D, F = unpack_vars(z)
    return np.sum(X[0,:]) - 1.0

def depot_return(z):
    X, D, F = unpack_vars(z)
    return np.sum(X[:,0]) - 1.0

def visit_customer(j):
    def c(z):
        X, D, F = unpack_vars(z)
        return np.sum(X[:,j]) - 1.0
    return c

def leave_customer(i):
    def c(z):
        X, D, F = unpack_vars(z)
        return np.sum(X[i,:]) - 1.0
    return c

def no_loop(i):
    def c(z):
        X, D, F = unpack_vars(z)
        return -X[i,i]
    return c

# load & energy constraints

def depot_load_zero(z):
    X, D, F = unpack_vars(z)
    return D[0]   # =0

def depot_energy_zero(z):
    X, D, F = unpack_vars(z)
    return F[0]   # =0

# remaining load & energy constraints enforced via bounds
bounds = []

bounds.extend([(0,1)] * num_x)       # X_ij
bounds.extend([(0,Q)] * num_D)       # D_i
bounds.extend([(0,Emax)] * num_F)    # F_i

# build constraints list
constraints = []
constraints.append({"type":"eq", "fun": depot_depart})
constraints.append({"type":"eq", "fun": depot_return})
constraints.append({"type":"eq", "fun": depot_load_zero})
constraints.append({"type":"eq", "fun": depot_energy_zero})

for j in range(1, N):
    constraints.append({"type":"eq", "fun": visit_customer(j)})
for i in range(1, N):
    constraints.append({"type":"eq", "fun": leave_customer(i)})

for i in range(N):
    constraints.append({"type":"ineq", "fun": no_loop(i)})

# initial guess

X0 = np.zeros((N,N))
X0[0,1] = 1
X0[1,2] = 1
X0[2,3] = 1
X0[3,0] = 1

z0 = np.zeros(num_vars)
z0[:num_x] = X0.flatten()

solution = minimize(
    objective,
    z0,
    bounds=bounds,
    constraints=constraints,
)

print("Success:", solution.success)
print("Message:", solution.message)
print("Total cost:", solution.fun)

# X: routing matrix, D: load vector, F: energy vector
X_opt, D_opt, F_opt = unpack_vars(solution.x)

print("X:\n", X_opt)
print("D:", D_opt)
print("F:", F_opt)
