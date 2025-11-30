import numpy as np
import pulp as pl
import math

# node data (simulation values from paper)
# depot + customers
node_coords = {
    0: (35, 35),  # depot
    1: (40.356, 36.287),
    2: (29.768, 33.453),
    3: (36.925, 27.514),
    4: (31.873, 39.642),
    5: (38.597, 31.365),
    6: (34.639, 41.789),
    7: (32.218, 33.891),
    8: (37.462, 39.127),
    9: (30.853, 36.745),
    10: (38.934, 30.186)
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
        d[i, j] = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)

WIND_DIRECTION = 40 # (0 - 360) degrees from north

def direction_multiplier(direction_deg, wind_deg,
                         max_factor=1.5, min_factor=0.5):
    # Smallest angle difference between directions, in [0, 180]
    diff = abs((direction_deg - wind_deg + 180) % 360 - 180)

    # Alignment in [-1, 1]: 1 = same direction, -1 = opposite
    alignment = math.cos(math.radians(diff))

    # Map alignment [-1, 1] to [min_factor, max_factor]
    base = (max_factor + min_factor) / 2.0   # 1.0 for 0.8–1.2
    scale = (max_factor - min_factor) / 2.0  # 0.2 for 0.8–1.2

    return base + alignment * scale

def angle_from_north(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # atan2(dx, dy) instead of atan2(dy, dx) gives angle relative to +Y (north)
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    # Normalize to [0, 360)
    return (angle_deg + 360) % 360

# direction headwind / tailwind co-efficient
# > 1 == tailwind
# < 1 == headwind
multiplier = np.zeros((N, N))
for i in node_ids:
    xi, yi = node_coords[i]
    for j in node_ids:
        xj, yj = node_coords[j]
        direction = angle_from_north(xi, yi, xj, yj)
        multiplier[i, j] = direction_multiplier(direction, WIND_DIRECTION)

print(multiplier)
# no fly zone coordinates
# changes optimized path: 0 -> 2 -> 1 -> 3 -> 0
no_fly_polygon = np.array([
    (37.247, 36.096),
    (37.330, 35.927),
    (37.856, 35.666),
    (37.496, 36.112),
])

# line-polygon intersection check
def ccw(A, B, C):
    # returns True if the three points make a counter-clockwise turn.
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(P1, P2, Q1, Q2):
    # returns True if segment P1->P2 intersects segment Q1->Q2
    return ccw(P1, Q1, Q2) != ccw(P2, Q1, Q2) and \
           ccw(P1, P2, Q1) != ccw(P1, P2, Q2)

def segment_intersects_polygon(P1, P2, poly):
    # check all edges of a polygon
    for k in range(len(poly)):
        Q1 = poly[k]
        Q2 = poly[(k + 1) % len(poly)]
        if segments_intersect(P1, P2, Q1, Q2):
            return True   
    return False

# build arc validity matrix
invalid_arc = np.zeros((N, N), dtype=bool)

for i in node_ids:
    for j in node_ids:
        if i == j:
            invalid_arc[i, j] = True
            continue

        P1 = node_coords[i]
        P2 = node_coords[j]

        if segment_intersects_polygon(P1, P2, no_fly_polygon):
            invalid_arc[i, j] = True

# === milp model with pulp ===

c3 = 0.66  # cost per unit of energy consumption (from paper)

# discrete speed levels (m/s) to approximate continuous v in a milp
speed_levels = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
S = range(len(speed_levels))

# precompute valid arcs
valid_arcs = [(i, j) for i in node_ids for j in node_ids if not invalid_arc[i, j]]

# precompute energy and cost coefficients for each (i,j,s)
energy = {}  # kwh for traversing arc (i,j) at speed s
cost = {}    # cost contribution c3 * energy
for (i, j) in valid_arcs:
    dist_km = d[i, j]               # assume distance in km
    for s_idx, v in enumerate(speed_levels):
        time_s = (dist_km * 1000.0) / v      # seconds
        E_ij_s = (P_const * time_s) / (3.6e6 * multiplier[(i, j)]) # kwh

        energy[(i, j, s_idx)] = E_ij_s
        cost[(i, j, s_idx)] = c3 * E_ij_s

# create milp problem
model = pl.LpProblem("UAV_Route_MILP", pl.LpMinimize)

# decision variables:
# x[i,j] = 1 if uav uses arc (i,j), 0 otherwise
x = {
    (i, j): pl.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
    for (i, j) in valid_arcs
}

# y[i,j,s] = 1 if uav uses arc (i,j) at discrete speed level s
y = {
    (i, j, s): pl.LpVariable(f"y_{i}_{j}_{s}", lowBound=0, upBound=1, cat="Binary")
    for (i, j) in valid_arcs
    for s in S
}

# MTZ ordering variables u[i]
u = {
    i: pl.LpVariable(f"u_{i}", lowBound=0, upBound=N - 1, cat="Continuous")
    for i in node_ids
}

# objective: minimize power consumption cost (same structure as before)
model += pl.lpSum(cost[i, j, s] * y[(i, j, s)] for (i, j) in valid_arcs for s in S)

# === constraints ===

# tie x and y: if arc (i,j) is used, exactly one speed is chosen
for (i, j) in valid_arcs:
    model += pl.lpSum(y[(i, j, s)] for s in S) == x[(i, j)], f"speed_choice_{i}_{j}"

# departure starting point of the uavs is the uav depot (eq 14)
model += pl.lpSum(x.get((0, j), 0) for j in node_ids if (0, j) in x) == 1, "depot_depart"

# uavs eventually return to the uav depot (eq 15)
model += pl.lpSum(x.get((i, 0), 0) for i in node_ids if (i, 0) in x) == 1, "depot_return"

# each customer node is visited only once (eq 13) as incoming degree = 1
for j in range(1, N):
    model += pl.lpSum(x.get((i, j), 0) for i in node_ids if (i, j) in x) == 1, f"visit_once_{j}"

# outgoing degree constraints (including depot, matching original logic)
for i in node_ids:
    model += pl.lpSum(x.get((i, j), 0) for j in node_ids if (i, j) in x) == 1, f"outgoing_degree_{i}"

# fix_depot_order: u[0] = 0
model += u[0] == 0, "fix_depot_order"

# mtz subtour elimination constraints for customers (i,j > 0, i != j)
for i in range(1, N):
    for j in range(1, N):
        if i == j:
            continue
        if (i, j) in x:
            # u[i] - u[j] + N*x[i,j] <= N - 1
            model += u[i] - u[j] + N * x[(i, j)] <= N - 1, f"mtz_{i}_{j}"

# battery constraint: total energy consumption <= emax (kwh)
model += pl.lpSum(energy[(i, j, s)] * y[(i, j, s)]
                  for (i, j) in valid_arcs for s in S) <= Emax, "battery_limit"

# solve the milp
status = model.solve()  # uses cbc by default if installed

print("Status:", pl.LpStatus[status])
print("Objective (Energy cost):", pl.value(model.objective))

# recover x and speeds v, plus ordering u
X_opt = np.zeros((N, N))
V_opt = np.zeros((N, N))

for (i, j) in valid_arcs:
    x_val = pl.value(x[(i, j)])
    if x_val is None:
        x_val = 0.0
    X_opt[i, j] = x_val

    # reconstruct speed as expected value over discrete choices
    speed_val = 0.0
    for s_idx, v in enumerate(speed_levels):
        y_val = pl.value(y[(i, j, s_idx)])
        if y_val is None:
            y_val = 0.0
        speed_val += v * y_val
    V_opt[i, j] = speed_val

u_opt = np.array([pl.value(u[i]) for i in node_ids])

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
