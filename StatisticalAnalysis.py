import numpy as np
import matplotlib.pyplot as plt

def compute_vacf(vx_all, vy_all):
    n_frames, n_particles = vx_all.shape
    

    acf_total = np.zeros(n_frames)

    divisor = np.arange(n_frames, 0, -1)

    for p in range(n_particles):
        vx = vx_all[:, p]
        vy = vy_all[:, p]
        
        # np.correlate(mode='full') returns lags from -(N-1) to (N-1). We slice [n_frames-1:] to get the positive lags 0 to N-1.
        
        vel_corr = np.correlate(vx, vx, mode='full')[n_frames-1:] + np.correlate(vy, vy, mode='full')[n_frames-1:]
        vel_corr = vel_corr /divisor
        
        if vel_corr[0] != 0:
            acf_total += (vel_corr / vel_corr[0])

    # Average over all particles
    acf_avg = acf_total / n_particles
    
    return acf_avg

def compute_gr(xi_a, yi_a, dr = 0.001,r1 = 0.05,r2 = 0.5, L = 10):
    # L = 10
    dists_1000 = []
    bins = np.arange(r1,r2,dr)
    bins_cen = (bins[0:-1] + bins[1:])/2
    for i in range(500,800):
        xi = xi_a[i];yi = yi_a[i]
        dx = np.abs(xi[:, None] - xi[None, :])
        dx = np.minimum(dx, L - dx) 
            
        dy = np.abs(yi[:, None] - yi[None, :])
        dy = np.minimum(dy, L - dy)
            
        dist_matrix = np.sqrt(dx**2 + dy**2)

        dists_1000 = np.append(dists_1000, np.triu(dist_matrix).flatten())
    
    hist, bins = np.histogram(dists_1000, bins = bins, density=True)
    hist= hist/(2*np.pi*bins_cen*dr)

    return bins_cen, hist
