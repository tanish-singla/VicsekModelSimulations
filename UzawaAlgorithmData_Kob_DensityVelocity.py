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


## Initialize the simulation with  all parameters
n = 1200
L = 10
r0 = 0.35
v = 0.005 # Desired speed (step size per frame)
dt = 1.0 
rp1 = 0.15
rp2 = 0.10
xi, yi, rp = generate_random_packing(n,L,rp1,rp2)
theta = np.random.rand(n) * 2 * np.pi - np.pi

def get_distances_pbc(x, y, L):
    """
    This function calculates the periodic distance matrix
    """
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    dist = np.sqrt(dx**2 + dy**2)
    return dx, dy, dist

def solve_uzawa_velocity(xi, yi, theta, v_speed, rp, L, max_iter=10000, rho=0.2):
    """
    Adjusts velocities to strictly enforce non-overlap using Uzawa projection.
    
    rho: Learning rate/step size for the pressure projection.
    """
    n = len(xi)
    
    #  Desired Velocities (Vicsek term)
    # U is the velocity the particles want to have
    ux = v_speed * np.cos(theta)
    uy = v_speed * np.sin(theta)

    vx = ux.copy()
    vy = uy.copy()
    
    # 2. Identify Contacts 
    dx, dy, dist = get_distances_pbc(xi, yi, L)
    
    # Radii sum matrix
    r_sum = rp[:, None] + rp[None, :]
    
    # Find pairs that might collide: distance < sum of radii
    contact_mask = (dist < r_sum) & (dist > 0)
    
    # Extract indices of interacting pairs to avoid N^2 loops in the solver
    # We only take the upper triangle to avoid double counting pairs (i,j) and (j,i)
    pairs_i, pairs_j = np.where(np.triu(contact_mask, k=1))
    
    if len(pairs_i) == 0:
        return vx, vy 

    # Normal vector n_ij points from j to i (pushing i away)
    nx = dx[pairs_i, pairs_j] / (dist[pairs_i, pairs_j] + 1e-9)
    ny = dy[pairs_i, pairs_j] / (dist[pairs_i, pairs_j] + 1e-9)
    
    # Gap: How much overlap? (Negative value = overlap amount)
    # We want dist >= r_sum, so gap = dist - r_sum. 
    gaps = dist[pairs_i, pairs_j] - r_sum[pairs_i, pairs_j]
    
    # Pressures initialized to 0
    # One lambda per contact pair
    lambdas = np.zeros(len(pairs_i))
    
    # 3. Uzawa Iteration Loop
    for iteration in range(max_iter):
        # A. Update Velocities based on current Pressures (Lambdas)
        # Force on i from j is lambda * normal. Force on j is opposite.t
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Vectorized force accumulation
        # Force on i: +lambda * n (pushed away from j)
        np.add.at(fx, pairs_i, lambdas * nx)
        np.add.at(fy, pairs_i, lambdas * ny)
        # Force on j: -lambda * n (pushed away from i)
        np.add.at(fx, pairs_j, -lambdas * nx)
        np.add.at(fy, pairs_j, -lambdas * ny)
        
        # V = U + Force (Assuming mass = 1 and dt included in rho effectively)
        # In exact projection: V = U - B.T * lambda
        vx = ux + fx
        vy = uy + fy
        
        # Update Pressures (Project constraints)
        # Calculate relative velocity along normal: (v_i - v_j) . n
        # Note: We defined n pointing j->i. 
        # Closing speed is negative if they move closer.
        dvx = vx[pairs_i] - vx[pairs_j]
        dvy = vy[pairs_i] - vy[pairs_j]
        
        v_rel = dvx * nx + dvy * ny
        
        # Constraint violation: 
        # We want: v_rel + gap/dt >= 0
        # If gap is negative (overlap), we need positive v_rel (separation).
        constraint = v_rel + (gaps / dt)
        
        # Update lambda: increase pressure if constraint violated (< 0)
        # lambda_{k+1} = max(0, lambda_k - rho * constraint)
        # Note: The sign depends on formulation. If constraint < 0 is bad:
        lambdas = np.maximum(0, lambdas - rho * constraint)
        
        # Optional: Check convergence
        if np.max(np.abs(rho * constraint)) < 1e-3: break
        if iteration==max_iter-1: return np.where(fx == 0, vx, 0),np.where(fy == 0, vy, 0)
    # print(iteration)
    return vx, vy


def update():
    global xi, yi, theta, smoothed_speeds
    
    # 1. Vicsek Alignment (Orientation Update)
    dx, dy, dist = get_distances_pbc(xi, yi, L)
    np.fill_diagonal(dist, np.inf)
    
    neighbors = dist < r0
    
    # Calculate average direction of neighbors
    sin_sum = np.sum(np.sin(theta)[None, :] * neighbors, axis=1)
    cos_sum = np.sum(np.cos(theta)[None, :] * neighbors, axis=1)
    counts = np.sum(neighbors, axis=1)
    
    mask = counts > 0
    new_theta = theta.copy()
    # Add noise to orientation
    noise = 0.5 * np.random.randn(np.sum(mask))
    new_theta[mask] = np.arctan2(sin_sum[mask], cos_sum[mask]) + noise
    theta = new_theta

    vx, vy = solve_uzawa_velocity(xi, yi, theta, v, rp, L)
    vx, vy = np.where(counts>5,vx/10,vx), np.where(counts>5,vy/10,vy)
    xi += vx * dt 
    yi += vy * dt
    
    #  Enforce PBC on positions
    xi = xi % L
    yi = yi % L


    v_2 = vx**2 + vy**2

    return vx,vy, np.sqrt(np.mean(v_2))
    



if __name__ == '__main__':

    data = {}
    data['rp'] = rp
    data['xi'] = [xi]
    data['yi'] = [yi]
    data['vx'] = [v*np.cos(theta)]
    data['vy'] = [v*np.sin(theta)]
    data['v_rms'] = [v]

    for i in range(3600):
        vx,vy, v_rms = update()
        data['xi'].append(xi)
        data['yi'].append(yi)
        data['vx'].append(vx)
        data['vy'].append(vy)
        data['v_rms'].append(v_rms)


    np.savez(f'{n}pts_0pt15_0pt10_size_0pt35ro_uzawa_DD_0pt85A.npz', **data)

