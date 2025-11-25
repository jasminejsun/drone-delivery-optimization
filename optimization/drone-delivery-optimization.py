import numpy as np
from scipy.optimize import minimize

# node data (simulation values from paper)
# depot + 3 customers
node_coords = {
    0: (35, 35),  # depot
    1: (40.356, 36.287),
    2: (29.768, 33.453),
    3: (36.925, 27.514),
}

N = len(node_coords)
node_ids = list(node_coords.keys())

# energy constants (from paper)
alpha = 101.18
beta  = -1381.73
wu = 36           # UAV mass (kg)

# fixed payload mass which decreases negligibly per stop (assumption)
payload = 2.0     

# battery capacity (usable kWh)
Emax = 4.0        

# the paper uses a linearized model: P = alpha * (w_u + q) + beta which gives propulsion power in Watts.
def power_linear(q):
    return alpha * (wu + q) + beta

P_const = power_linear(payload)

# distance matrix
d = np.zeros((N, N))
for i in node_ids:
    xi, yi = node_coords[i]
    for j in node_ids:
        xj, yj = node_coords[j]
        d[i,j] = np.sqrt((xj - xi)**2 + (yj - yi)**2)


# decision variables
# X[i,j] = routing decision: whether UAV flies from node i to node j.
# V[i,j] = cruise speed on arc (i,j) in m/s
num_x = N * N # number of X[i,j] variables
num_v = N * N # number of V[i,j] variables
num_u = N     # MTZ ordering variables

# no fly zone coordinates
# optimized path: 0 -> 1 -> 2 -> 3 -> 0
# no_fly_polygon = np.array([
#     (30, 40),
#     (31, 42),
#     (34, 41),
#     (31, 39)
# ])

# changes optimized path: 0 -> 2 -> 1 -> 3 -> 0
no_fly_polygon = np.array([
    (37.247, 36.096),
    (37.330, 35.927),
    (37.856, 35.666),
    (37.496, 36.112),
])

# Drops the path because of our hard bounds
# no_fly_polygon = np.array([
#     (36.5, 34.8),
#     (37.6, 36.4),
#     (39.2, 35.9),
#     (38.1, 34.2),
# ])

# line-polygon intersection check
def ccw(A, B, C):
    # returns True if the three points make a counter-clockwise turn.
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def segments_intersect(P1, P2, Q1, Q2):
    # returns True if segment P1->P2 intersects segment Q1->Q2
    return ccw(P1, Q1, Q2) != ccw(P2, Q1, Q2) and \
           ccw(P1, P2, Q1) != ccw(P1, P2, Q2)

def segment_intersects_polygon(P1, P2, poly):
    # check all edges of a polygon
    for k in range(len(poly)):
        Q1 = poly[k]
        Q2 = poly[(k+1) % len(poly)]
        if segments_intersect(P1, P2, Q1, Q2):
            return True
    return False

# build arc validity matrix
invalid_arc = np.zeros((N, N), dtype=bool)

for i in node_ids:
    for j in node_ids:
        if i == j:
            invalid_arc[i,j] = True
            continue

        P1 = node_coords[i]
        P2 = node_coords[j]

        if segment_intersects_polygon(P1, P2, no_fly_polygon):
            invalid_arc[i,j] = True

def unpack(z):
    X = z[:num_x].reshape((N,N))
    V = z[num_x:num_x + num_v].reshape((N,N))
    u = z[num_x + num_v:]
    return X, V, u

c3 = 0.66 # cost per unit of energy consumption (from paper)

def objective(z):
    X, V, _ = unpack(z)

    time = (d * 1000) / V            # seconds
    f_ij = (P_const * time) / 3.6e6  # energy consumed by UAV k on arc (i,j) kWh

    F3 = c3 * np.sum(f_ij * X) # function related to power consumption cost
    return F3


# routing constraints

# departure starting point of the UAVs is the UAV depot (Eq 14)
def depot_depart(z):
    X, _, _ = unpack(z)
    return np.sum(X[0,:]) - 1

# UAVs eventually return to the UAV depot (Eq 15)
def depot_return(z):
    X, _, _ = unpack(z)
    return np.sum(X[:,0]) - 1

# each customer node is visited only once (Eq 13)
def visit_once(j):
    def c(z):
        X, _, _ = unpack(z)
        return np.sum(X[:,j]) - 1
    return c

# prevents the UAV from traveling from node i back to itself
def no_loop(i):
    def c(z):
        X, _, _ = unpack(z)
        return -X[i,i]
    return c

def outgoing_degree(i):
    def c(z):
        X, _, _ = unpack(z)
        return np.sum(X[i,:]) - 1
    return c

def fix_depot_order(z):
    _, _, u = unpack(z)
    return u[0]

def mtz_constraint(i, j):
    def c(z):
        X, _, u = unpack(z)
        return (N - 1) - (u[i] - u[j] + N * X[i,j])
    return c


# battery constraint
# cumulative power consumption of the UAV must not exceed the maximum capacity of the UAV's batteries (Eq 26)
def battery_limit(z):
    X, V, _ = unpack(z)
    time = (d * 1000) / V
    E_ij = (P_const * time) / 3.6e6
    return Emax - np.sum(E_ij * X)    # must be >= 0

bounds = []

# bound routing variables X[i,j] between 0 and 1
for i in range(N):
    for j in range(N):
        if invalid_arc[i,j]:
            bounds.append((0, 0))     # X[i,j] must be 0
        else:
            bounds.append((0, 1))     # normal arc

# cruise speed allowed between 10 m/s and 20 m/s (assumption)
speed_min = 10
speed_max = 20
bounds.extend([(speed_min, speed_max)] * num_v)

# MTZ ordering variable bounds
bounds.extend([(0, N - 1)] * num_u)


# initial guess
# route 0→1→2→3→0
X0 = np.zeros((N,N))
X0[0,1] = 1
X0[1,2] = 1
X0[2,3] = 1
X0[3,0] = 1

# initial speed = 14 m/s (assumption)
V0 = np.ones((N,N)) * 14
u0 = np.arange(N)
u0[0] = 0

z0 = np.concatenate([X0.flatten(), V0.flatten(), u0])


constraints = []
constraints.append({"type": "eq", "fun": depot_depart})
constraints.append({"type": "eq", "fun": depot_return})
constraints.append({"type": "ineq", "fun": battery_limit})

for j in range(1, N):
    constraints.append({"type":"eq", "fun": visit_once(j)})

for i in range(N):
    constraints.append({"type":"ineq", "fun": no_loop(i)})

for i in range(1, N):
    constraints.append({"type":"eq", "fun": outgoing_degree(i)})

constraints.append({"type":"eq", "fun": fix_depot_order})

for i in range(1, N):
    for j in range(1, N):
        if i == j:
            continue
        constraints.append({"type":"ineq", "fun": mtz_constraint(i, j)})

solution = minimize(
    objective,
    z0,
    bounds=bounds,
    constraints=constraints,
)

print("Success:", solution.success)
print("Message:", solution.message)
print("Objective (Energy, kWh):", solution.fun)

X_opt, V_opt, u_opt = unpack(solution.x)
print("Optimal Routing Matrix X:\n", X_opt)
print("Optimal Speeds V (m/s):\n", V_opt)
print("Ordering variables u:\n", u_opt)

def extract_tour_from_X(X, start=0, tol=0.5):
    tour = [start]
    current = start
    visited = {start}
    max_iters = X.shape[0] * X.shape[0]

    for _ in range(max_iters):
        row = X[current]
        next_idx = int(np.argmax(row))
        if row[next_idx] <= tol:
            break
        tour.append(next_idx)
        if next_idx == start:
            break
        if next_idx in visited:
            break
        visited.add(next_idx)
        current = next_idx
    return tour

tour = extract_tour_from_X(X_opt)
tour_str = " -> ".join(map(str, tour))
print(f"Recovered tour: {tour_str}")
