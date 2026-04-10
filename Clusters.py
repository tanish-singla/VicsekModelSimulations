## Contains the functions to identify clusters using DBSCAN, fit ellipses using PCA and make videos from data
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import random

def identify_clusters_dbscan(xi, yi, vx, vy, L=10.0, eps_r=0.35, eps_v=0.001, min_pts=2):
    """
    eps_r: maximum distance between particles to be qualified to be in a cluster
    eps_v: Maximum  magnitude of difference in velocity vectors to be in a cluster
    min_pts: Minimum number of qualified points around one qualifed point to be considered in a cluster
    """
    n_particles = len(xi)
    
    # Distance (PBC)
    dx = np.abs(xi[:, None] - xi[None, :])
    dx = np.minimum(dx, L - dx) 
    
    dy = np.abs(yi[:, None] - yi[None, :])
    dy = np.minimum(dy, L - dy)
    
    dist_matrix_r = np.sqrt(dx**2 + dy**2)
    
    # 2. Velocity Difference (Kinematic distance)
    dvx = vx[:, None] - vx[None, :]
    dvy = vy[:, None] - vy[None, :]
    dist_matrix_v = np.sqrt(dvx**2 + dvy**2)
    
    norm_r = dist_matrix_r / eps_r
    norm_v = dist_matrix_v / eps_v
    combined_distance_matrix = np.maximum(norm_r, norm_v)
    
    db = DBSCAN(eps=1.0, min_samples=min_pts, metric='precomputed')
    labels = db.fit_predict(combined_distance_matrix)

    cluster_ids, sizes = np.unique(labels, return_counts=True)
    
    # We ignore the -1 label, because those are isolated noise particles, not a true cluster.
    n_clusters = len(cluster_ids[cluster_ids != -1])
    
    return labels, n_clusters, cluster_ids, sizes

def run_cluster_identification(N, dd, fname):
    """
    N : Array of different number of particles (proxy for different densities) to run this alogorithm for
    dd: Directory from which to read simulation data and in which to save all results
    fname: Filename to read and save all data in. The file name will automatically be prefixed by the number of particles and saved
    """
    #N = [400,600, 800, 1000, 1200, 1400]
    # dd = "UzawaSimulations_65Kob_0pt07_DynamicVelocity/"
    # fname = "pts_0pt15_0pt07_size_0pt35ro_uzawa_DD_0pt65A"
    for n in N:
        data = np.load(dd+f"{n}" + fname + ".npz")
        xi_a,yi_a,vx_a,vy_a =  data['xi'], data['yi'], data['vx'], data['vy']
        labels, n_clusters, cluster_ids, sizes =  [], [], [], []
        for xi, yi, vx, vy in zip(xi_a,yi_a,vx_a,vy_a):
            res = identify_clusters_dbscan(xi, yi, vx, vy)
            labels.append(res[0])
            n_clusters.append(res[1])
            cluster_ids.append([res[2]])
            sizes.append(res[3])

        d =  {}
        cluster_ids_concat = np.concatenate([cluster_ids[i][0] for i in range(len(cluster_ids))])
        sizes_concat = np.concatenate(sizes)
        d["labels"] = labels; d["n_clusters"] = n_clusters; d["cluster_ids"] =cluster_ids_concat; d["sizes"] = sizes_concat
        np.savez(dd + f"{n}" + fname + "_clustersDBSCAN.npz", **d)

def fit_ellipse_PCA(cx, cy, L=10.0, scale=2.5):
    """
    Fits the ellipse using PCA. Calculates the covariance matrix of x and y coordinates 
        of a cluster. Finds the eigenvectors and eigenvalues. The eigenvectors will always
        be perpendicular to each other as covariance matrix is symmetric. The eigenvalues 
        could be treated as axes lengths with proper scaling to include maximum number of particles

    cx, cy: Arrays of x and y coordinates of particles IN THE SAME CLUSTER.
    L: Box size.
    scale: How many standard deviations. 
           2.0 covers ~86% of points, 3.0 covers ~99%.
    """

    if len(cx) < 3:
        return None# Cannot fit a meaningful 2D ellipse to 1 or 2 points
    
    ref_x, ref_y = cx[0], cy[0]
    
    # Find distance from anchor to all other points using Minimum Image Convention
    dx = cx - ref_x
    dy = cy - ref_y
    dx = dx - L * np.round(dx / L)
    dy = dy - L * np.round(dy / L)
    # Reconstruct continuous coordinates that ignore the boundary break
    unwrapped_x = ref_x + dx
    unwrapped_y = ref_y + dy
    #Find the Center
    mean_x = np.mean(unwrapped_x)
    mean_y = np.mean(unwrapped_y)
    
    #Calculate Covariance Matrix
    cov = np.cov(unwrapped_x, unwrapped_y)
    
    # Get Eigenvalues and Eigenvectors
    evals, evecs = np.linalg.eigh(cov)
    
    # Sort them so the largest eigenvalue (major axis) is first
    order = evals.argsort()[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
 
    angle_rad = np.arctan2(evecs[1, 0], evecs[0, 0])
    angle_deg = np.degrees(angle_rad)

    width = 2 * scale * np.sqrt(np.maximum(evals[0], 0))
    height = 2 * scale * np.sqrt(np.maximum(evals[1], 0))
    
    mean_x = mean_x % L
    mean_y = mean_y % L

    return mean_x, mean_y, width, height, angle_deg

def create_video_with_ellipse_fitting(N):

    # N = [400,600, 800, 1000, 1200, 1400]
    for n in N:
        L = 10
        dd = "UzawaSimulations_65Kob_0pt07_DynamicVelocity/"
        data = np.load(dd+f"{n}pts_0pt15_0pt07_size_0pt35ro_uzawa_DD_0pt65A.npz")
        data_clusters = np.load(dd+f"{n}pts_0pt15_0pt07_size_0pt35ro_uzawa_DD_0pt65A_clustersDBSCAN.npz")
        
        vx = data['vx'][0]
        vy = data['vy'][0]
        xi, yi = data['xi'][0], data['yi'][0]
        
        v_actual = np.sqrt(vx**2 + vy**2)

        ### Smoothing the visualization (video) only to prevent sudden flashes.
        smoothed_speeds = v_actual
        alpha = 0.25
        ##

        cmap_speed = cm.viridis
        norm_speed = mcolors.Normalize(vmin=0, vmax=0.005)
        # cmap_cluster = cm.flag
        # norm_cluster = mcolors.Normalize(3,n/5)

        counter = 500 #video starting frame
        master_ellipse_data = []
        
        fig, ax = plt.subplots(figsize=(7, 6))

        # Colorbar for speed
        sm = cm.ScalarMappable(cmap=cmap_speed, norm=norm_speed)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Particle Speed', rotation=270, labelpad=15)

        # Global cache for cluster colors
        global_cluster_colors = {}

        def get_cluster_color(cid,sizes):
            if cid not in global_cluster_colors:
                if cid==-1:
                    global_cluster_colors[cid] = 'yellow'
                # random.seed(int(cid))
                # global_cluster_colors[cid] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                else:
                    random.seed(int(sizes[cid]))
                    global_cluster_colors[cid] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                    # global_cluster_colors[cid] = cmap_cluster(norm_cluster(sizes[cid]))
            return global_cluster_colors[cid]
        
        def update(frame):
            global smoothed_speeds, counter, master_ellipse_data
            vx = data['vx'][counter]
            vy = data['vy'][counter]
            xi, yi = data['xi'][counter], data['yi'][counter]
            rp  = data['rp']
            labels = data_clusters['labels'][counter]
            n_clusters = data_clusters['n_clusters'][counter]
            n_clusters_prev = np.sum(data_clusters['n_clusters'][0:counter])
            sizes = data_clusters['sizes'][n_clusters_prev:n_clusters_prev+n_clusters]
            
            v_actual = np.sqrt(vx**2 + vy**2)
            smoothed_speeds = (alpha * v_actual) + ((1.0 - alpha) * smoothed_speeds)

            ax.clear()

            for x, y, speed, cid, ri in zip(xi, yi, smoothed_speeds, labels, rp):
                # 1. Cluster Halo (The visual boundary)
                if cid != -1: # Assuming -1 is noise/unclustered
                    cluster_color = get_cluster_color(cid, sizes)
                    # Draw a larger, semi-transparent circle behind the particle
                    halo_radius = ri * 2.5 # Tweak this multiplier to control how much the blobs merge
                    ax.add_patch(patches.Circle((x, y), radius=halo_radius, 
                                                facecolor=cluster_color, edgecolor='none', 
                                                alpha=0.35, zorder=1))

                # 2. Actual Particle (Colored by speed)
                fill_color = cmap_speed(norm_speed(speed))
                ax.add_patch(patches.Circle((x, y), radius=ri, facecolor=fill_color, 
                                            edgecolor='black', lw=0.5, alpha=1.0, zorder=2))
            for cid in np.unique(labels):
                if cid==-1:continue
                cx = xi[labels==cid]
                cy = yi[labels==cid]
                stats = fit_ellipse_PCA(cx,cy)
                if stats is not None:
                    mean_x, mean_y, width, height, angle_deg = stats

                    master_ellipse_data.append([
                            counter, cid, mean_x, mean_y, width, height, angle_deg, len(cx)
                        ])

                    ell = Ellipse(xy=(mean_x, mean_y),
                        width=width, height=height,
                        angle=angle_deg,
                        edgecolor= get_cluster_color(cid, sizes), facecolor=get_cluster_color(cid, sizes),
                        alpha=1, lw=2, zorder=1)
                    ax.add_patch(ell)



            ax.quiver(xi, yi, vx, vy, color='darkslategray', scale=0.2, zorder=3)

            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_aspect('equal')
            
            counter += 1

        ani = FuncAnimation(fig, update, frames=50, interval=40, blit=False)

        w = PillowWriter(fps=5)
        ani.save(dd+f"{n}pts_0pt15_0pt07_size_0pt35ro_uzawa_DD_0pt65A_DBSCAN_ellipse.gif", w)
        ellipse_array = np.array(master_ellipse_data)
        np.savez(dd + f"{n}pts_0pt15_0pt07_size_0pt35ro_uzawa_DD_0pt65A_DBSCAN_ellipse_data.npz",
            frames=ellipse_array[:, 0],
            cids=ellipse_array[:, 1],
            centers_x=ellipse_array[:, 2],
            centers_y=ellipse_array[:, 3],
            major_axes=ellipse_array[:, 4],
            minor_axes=ellipse_array[:, 5],
            angles=ellipse_array[:, 6],
            cluster_sizes=ellipse_array[:, 7])
