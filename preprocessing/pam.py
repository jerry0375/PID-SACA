"""PAM station-selection helper that consumes DBSCAN output and writes selected station coordinates."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Config:


    INPUT_FILE = REPO_ROOT / "data" / "preprocessing" / "dbscan" / "clustering_results.txt"
    OUTPUT_PATH = REPO_ROOT / "data" / "stations_30"
    PLOT_PATH = REPO_ROOT / "figure" / "preprocessing" / "pam"


    N_CLUSTERS = 30
    MAX_ITER = 300
    RANDOM_STATE = 42


    PAM_SAMPLE_SIZE = 20000


    FIGURE_SIZE = (14, 10)
    DPI = 150
    POINT_SIZE = 3
    POINT_ALPHA = 0.4
    CENTER_SIZE = 200
    CENTER_MARKER = '*'
    MAX_PLOT_POINTS = 50000

    @staticmethod
    def ensure_dir(path):

        Path(path).mkdir(parents=True, exist_ok=True)


def PAM(data, n_clusters=None, random_state=None, max_iter=None):


    if n_clusters is None:
        n_clusters = Config.N_CLUSTERS
    if random_state is None:
        random_state = Config.RANDOM_STATE
    if max_iter is None:
        max_iter = Config.MAX_ITER


    print("Extracting coordinates...")
    coordinates = np.array([[float(point[0]), float(point[1])] for point in data])

    print(f"\n{'='*60}")
    print(f"PAM Clustering Configuration")
    print(f"{'='*60}")
    print(f"Total data points: {len(coordinates):,}")
    print(f"Target stations: {n_clusters}")
    print(f"Sample size for PAM: {Config.PAM_SAMPLE_SIZE:,}")
    print(f"Estimated memory usage: ~{Config.PAM_SAMPLE_SIZE**2 * 8 / 1e9:.1f} GB")
    print(f"{'='*60}\n")


    if len(coordinates) > Config.PAM_SAMPLE_SIZE:
        print(f"[Step 1/4] Sampling {Config.PAM_SAMPLE_SIZE:,} representative points...")


        np.random.seed(random_state)
        sample_indices = np.random.choice(
            len(coordinates),
            Config.PAM_SAMPLE_SIZE,
            replace=False
        )
        sampled_coords = coordinates[sample_indices]
        print(f"  OK: Sampling completed: {len(sampled_coords):,} points")
    else:
        print(f"[Step 1/4] Dataset is small enough, using all points...")
        sampled_coords = coordinates
        print(f"  OK: Using all {len(sampled_coords):,} points")


    print(f"\n[Step 2/4] Running PAM algorithm on sampled data...")
    print(f"  This may take a few minutes...")

    kmedoids = KMedoids(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        method='pam',
        metric='euclidean',
        init='k-medoids++'
    )

    kmedoids.fit(sampled_coords)
    centers = kmedoids.cluster_centers_

    print(f"  OK: PAM converged after {kmedoids.n_iter_} iterations")
    print(f"  OK: Found {len(centers)} station locations (real data points)")


    print(f"\n[Step 3/4] Assigning all {len(coordinates):,} points to nearest stations...")


    batch_size = 100000
    labels = np.zeros(len(coordinates), dtype=int)

    for i in range(0, len(coordinates), batch_size):
        end = min(i + batch_size, len(coordinates))
        batch = coordinates[i:end]
        distances = cdist(batch, centers, metric='euclidean')
        labels[i:end] = np.argmin(distances, axis=1)

        if end < len(coordinates):
            print(f"  Progress: {end:,}/{len(coordinates):,} points processed")

    print("  OK: All points assigned to stations")


    print(f"\n[Step 4/4] Station statistics:")
    print(f"\n{'Station':<10} {'Passengers':<12} {'Percentage':<12} {'Location (Lon, Lat)'}")
    print(f"{'-'*70}")

    for i in range(n_clusters):
        count = np.sum(labels == i)
        percentage = count / len(labels) * 100
        lon, lat = centers[i]
        print(f"  {i+1:<8} {count:>10,} {percentage:>10.2f}%    ({lon:.6f}, {lat:.6f})")

    print(f"{'-'*70}")
    print(f"{'Total':<10} {len(labels):>10,} {100.0:>10.1f}%")

    return centers, labels, kmedoids


def visualize_clusters(data, centers, labels, save_path=None):


    Config.ensure_dir(Config.PLOT_PATH)


    coordinates = np.array([[float(point[0]), float(point[1])] for point in data])


    if len(coordinates) > Config.MAX_PLOT_POINTS:
        indices = np.random.choice(len(coordinates), Config.MAX_PLOT_POINTS, replace=False)
        plot_coords = coordinates[indices]
        plot_labels = labels[indices]
        sample_info = f" (showing {len(plot_coords)}/{len(coordinates)} points)"
    else:
        plot_coords = coordinates
        plot_labels = labels
        sample_info = ""


    fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE)


    colors = plt.cm.tab20(np.linspace(0, 1, Config.N_CLUSTERS))


    for i in range(Config.N_CLUSTERS):
        mask = plot_labels == i
        cluster_points = plot_coords[mask]
        if len(cluster_points) > 0:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=[colors[i]], s=Config.POINT_SIZE, alpha=Config.POINT_ALPHA,
                      edgecolors='none', label=f'Cluster {i+1}')


    ax.scatter(centers[:, 0], centers[:, 1],
              marker=Config.CENTER_MARKER, s=Config.CENTER_SIZE,
              c='red', edgecolors='black', linewidths=1,
              label='Bus Stations', zorder=1000)


    for i, center in enumerate(centers):
        ax.annotate(f'{i+1}', xy=(center[0], center[1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Bus Station Distribution (PAM Clustering){sample_info}\n'
                f'{Config.N_CLUSTERS} stations for {len(coordinates)} passengers',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)


    handles, labels_legend = ax.get_legend_handles_labels()

    station_handle = [h for h, l in zip(handles, labels_legend) if 'Station' in l]
    station_label = [l for l in labels_legend if 'Station' in l]
    if station_handle:
        ax.legend(station_handle, station_label, loc='upper right', fontsize=10, markerscale=0.5)

    if save_path is None:
        save_path = os.path.join(Config.PLOT_PATH, 'pam_clustering.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def output_station_locations(centers, save_path=None):


    Config.ensure_dir(Config.OUTPUT_PATH)

    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_PATH, 'bus_stations.txt')

    with open(save_path, 'w') as f:
        f.write("Station_ID,Longitude,Latitude\n")
        for i, center in enumerate(centers):
            f.write(f"{i+1},{center[0]:.6f},{center[1]:.6f}\n")

    print(f"Station locations saved to: {save_path}")


    return save_path


def read_file(file_path=None):


    data = []
    if file_path is None:
        file_path = Config.INPUT_FILE

    print(f"Reading data from: {file_path}")
    with open(file_path, 'r') as file:
        for line in file:

            if line.strip().split(',')[2] == '-1':
                continue
            data.append(line.strip().split(','))

    print(f"Loaded {len(data)} data points")
    return data

if __name__ == "__main__":
    print("=" * 70)
    print("Airport Shuttle Bus Station Planning System")
    print("Using PAM (K-Medoids) Clustering Algorithm")
    print("Finding REAL station locations from actual data points")
    print("=" * 70)


    print("\n[Phase 1] Reading denoised passenger drop-off data...")
    data = read_file()
    print(f"Sample data format: {data[0]}")


    print(f"\n[Phase 2] Finding optimal locations for {Config.N_CLUSTERS} bus stations...")
    centers, labels, model = PAM(data)


    print(f"\n[Phase 3] Generating visualization...")
    visualize_clusters(data, centers, labels)


    print(f"\n[Phase 4] Saving station locations...")
    output_station_locations(centers)


    print("\n" + "=" * 70)
    print("OK: PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total passengers analyzed: {len(data):,}")
    print(f"Bus stations identified: {Config.N_CLUSTERS}")
    print(f"Average passengers per station: {len(data) / Config.N_CLUSTERS:.0f}")
    print(f"\nAll stations are REAL DATA POINTS (actual passenger drop-off locations)")
    print(f"\nOutput files:")
    print(f"  - Visualization: {os.path.join(Config.PLOT_PATH, 'pam_clustering.png')}")
    print(f"  - Station list: {os.path.join(Config.OUTPUT_PATH, 'bus_stations.txt')}")
    print("=" * 70)
