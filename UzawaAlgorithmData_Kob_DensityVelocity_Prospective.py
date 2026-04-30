import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors


"""
All the simulation parameters and assumptions:
n: number of particles
L: box size
r0: Fixed Interaction Radius (Absolute Number not relative of particle size)
v: Desired particle speed
rp1: primary particle radius
rp2: secondary particle radius
R1A: primary particle number fraction
Initial angles are picked randomly from uniform distribution

Uzawa Alogrithm is a optimization algorithm. The final velocities will depend upon the
learning rate and max number of iterations allowed.

noise: noise to add to the average direction (currently 0.5*normal distribution which equates to +- 0.5 radians = +-30 degrees)

Density Dependent Velocity
if Number of neighbours in interaction radius (r0) > 5 then the velocity output by uzawa Algorithm drops by
a factor of 10
"""

def generate_random_packing(N, L, R1,R2, R1A = 0.85, max_iter=2000):
    """
    Generates a random packing of N particles of radius R 
    in an LxL box
    N: Number of Particles
    R1: Radius of Primary Particle
    R2: Radius of Secondary particle
    R1A: Number fraction of primary particle
    """
    # Theoretical Check: Can they even fit?
    # Max random packing fraction in 2D is ~0.82 to 0.84
    N1 =  int(R1A*N)
    N2 = N - N1
    area_fraction = (N1 * np.pi * R1**2) / (L**2) + (N2 * np.pi * R2**2) / (L**2)
    if area_fraction > 0.82:
        print(f"Warning: Density too high ({area_fraction:.2f}). Might not converge.")
        print("Max possible random packing is ~0.82.")

    x1 = np.random.uniform(R1, L - R1, N1)
    y1 = np.random.uniform(R1, L - R1, N1)

    x2 = np.random.uniform(R2, L - R2, N2)
    y2 = np.random.uniform(R2, L - R2, N2)
    
    x,y = np.concatenate((x1,x2)), np.concatenate((y1,y2))
    R= np.concatenate((np.full(N1, R1), np.full(N2, R2)))

    for iteration in range(max_iter):
        # Calculate pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Avoid self-interaction division by zero
        np.fill_diagonal(dist, np.inf)

        # Find overlaps (Target distance is 2*R)
        overlaps = R+R[:,None] - dist
        
        # Check if we are done (no overlaps > extremely small tolerance)
        if np.max(overlaps) < 1e-5:
            print(f"Converged in {iteration} iterations!")
            break

        # Get indices of overlapping pairs (only upper triangle to avoid double-correcting)
        mask = overlaps > 0
        pairs_i, pairs_j = np.where(np.triu(mask, k=1))

        # Push overlapping pairs apart
        for i, j in zip(pairs_i, pairs_j):
            overlap_amount = overlaps[i, j]
            
            # Normal direction
            nx = dx[i, j] / dist[i, j]
            ny = dy[i, j] / dist[i, j]
            
            # Move each particle halfway out of the overlap
            move_x = nx * overlap_amount * 0.5
            move_y = ny * overlap_amount * 0.5
            
            x[i] += move_x
            y[i] += move_y
            x[j] -= move_x
            y[j] -= move_y

        # Enforce Hard Wall Boundarie
        x = np.clip(x, R, L - R)
        y = np.clip(y, R, L - R)

    else:
        print(f"Reached max iterations ({max_iter}). Particles might still be slightly jammed.")

    return x, y, R

n = 1400
L = 10
r0 = 0.35
v = 0.005 
dt = 1.0 

rp1 = 0.07
rp2 = 0.10
xi, yi, rp = generate_random_packing(n,L,rp1,rp2, R1A = 0.75)
theta = np.random.rand(n) * 2 * np.pi - np.pi


def get_distances_pbc(x, y, L):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    dist = np.sqrt(dx**2 + dy**2)
    return dx, dy, dist

def solve_uzawa_velocity(xi, yi, theta, v_speed_array, rp, L, dt=1.0, max_iter=10000, rho=0.05):
    n = len(xi)

    ux = v_speed_array * np.cos(theta)
    uy = v_speed_array * np.sin(theta)

    # Proposed future positions
    xi_temp = xi + ux * dt
    yi_temp = yi + uy * dt

    dx, dy, dist = get_distances_pbc(xi_temp, yi_temp, L)
    r_sum = rp[:, None] + rp[None, :]

    contact_mask = (dist < r_sum) & (dist > 0)
    pairs_i, pairs_j = np.where(np.triu(contact_mask, k=1))

    if len(pairs_i) == 0:
        return ux, uy

    nx = dx[pairs_i, pairs_j] / (dist[pairs_i, pairs_j] + 1e-9)
    ny = dy[pairs_i, pairs_j] / (dist[pairs_i, pairs_j] + 1e-9)

    gaps = dist[pairs_i, pairs_j] - r_sum[pairs_i, pairs_j]
    lambdas = np.zeros(len(pairs_i))
    beta = 0.20

    # 3. Uzawa Iteration Loop
    for iteration in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        np.add.at(fx, pairs_i, lambdas * nx)
        np.add.at(fy, pairs_i, lambdas * ny)
        np.add.at(fx, pairs_j, -lambdas * nx)
        np.add.at(fy, pairs_j, -lambdas * ny)

        vx = ux + fx
        vy = uy + fy

        dvx = vx[pairs_i] - vx[pairs_j]
        dvy = vy[pairs_i] - vy[pairs_j]
        v_rel = dvx * nx + dvy * ny

        constraint = v_rel + (beta * gaps / dt)

        # Calculate new lambdas
        lambdas_new = np.maximum(0, lambdas - rho * constraint)

        if np.max(np.abs(lambdas_new - lambdas)) < 1e-7:
            lambdas = lambdas_new
            break

        lambdas = lambdas_new


    return vx, vy


def update():

    global xi, yi, theta

    # Vicsek Alignment (Orientation Update)
    dx, dy, dist = get_distances_pbc(xi, yi, L)
    np.fill_diagonal(dist, np.inf)

    neighbors = dist < r0

    # Calculate average direction of neighbors
    sin_sum = np.sum(np.sin(theta)[None, :] * neighbors, axis=1)
    cos_sum = np.sum(np.cos(theta)[None, :] * neighbors, axis=1)
    counts = np.sum(neighbors, axis=1)

    mask = counts > 0
    new_theta = theta.copy()
    noise = 0.5 * np.random.randn(np.sum(mask))
    new_theta[mask] = np.arctan2(sin_sum[mask], cos_sum[mask]) + noise
    theta = new_theta

    # Apply density slowdown to desired speed
    current_v_speed = np.where(counts > 5, v / 10, v)

    vx, vy = solve_uzawa_velocity(xi, yi, theta, current_v_speed, rp, L, dt=dt)

    max_allowed_speed = v * 2.0  # Cap at 2x the base desired speed

    actual_speeds = np.sqrt(vx**2 + vy**2)
    speed_violation_mask = actual_speeds > max_allowed_speed

    # If a particle exceeds the speed limit, scale its vector down to the limit
    with np.errstate(divide='ignore', invalid='ignore'):
        vx[speed_violation_mask] = (vx[speed_violation_mask] / actual_speeds[speed_violation_mask]) * max_allowed_speed
        vy[speed_violation_mask] = (vy[speed_violation_mask] / actual_speeds[speed_violation_mask]) * max_allowed_speed

    xi += vx * dt
    yi += vy * dt

    # Enforce PBC
    xi = xi % L
    yi = yi % L

    return vx, vy, np.sqrt(np.mean(vx**2 + vy**2))



if __name__ == '__main__':

    data = {}
    data['rp'] = rp
    data['xi'] = [xi]
    data['yi'] = [yi]
    data['vx'] = [v*np.cos(theta)]
    data['vy'] = [v*np.sin(theta)]
    data['v_rms'] = [v]

    for i in range(10000):
        vx,vy, v_rms = update()
        data['xi'].append(xi)
        data['yi'].append(yi)
        data['vx'].append(vx)
        data['vy'].append(vy)
        data['v_rms'].append(v_rms)


    np.savez(f'{n}pts_0pt15_0pt07_size_0pt35ro_uzawa_Prospective_DD_0pt75A.npz', **data)