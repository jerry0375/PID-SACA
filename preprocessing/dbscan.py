"""DBSCAN preprocessing helper for raw ride-order coordinates stored outside this repository."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Config:


    DATA_PATH = REPO_ROOT / "data" / "raw" / "total_ride_request"
    PLOT_PATH = REPO_ROOT / "figure" / "preprocessing" / "dbscan"
    OUTPUT_PATH = REPO_ROOT / "data" / "preprocessing" / "dbscan"


    EPS = 0.0009
    MIN_SAMPLES = 50


    START_DAY = 1
    END_DAY = 7


    K_NEIGHBORS = 50


    FIGURE_SIZE = (12, 5)
    DPI = 150
    POINT_SIZE = 2
    POINT_ALPHA = 0.6
    MAX_PLOT_POINTS = 100000

    @staticmethod
    def ensure_plot_dir():

        Path(Config.PLOT_PATH).mkdir(parents=True, exist_ok=True)


def dbscan_denoise(data, eps=None, min_samples=None):


    if eps is None:
        eps = Config.EPS
    if min_samples is None:
        min_samples = Config.MIN_SAMPLES


    data_array = np.array(data)


    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(data_array)


    mask = labels != -1
    denoised_data = data_array[mask]

    print(f"Original data points: {len(data_array)}")
    print(f"Noise points: {np.sum(labels == -1)}")
    print(f"Denoised data points: {len(denoised_data)}")
    print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")

    return denoised_data, data_array, labels


def plot_k_distance(data, k=None, save_path=None):


    if k is None:
        k = Config.K_NEIGHBORS

    Config.ensure_plot_dir()

    data_array = np.array(data)


    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(data_array)
    distances, indices = neighbors.kneighbors(data_array)


    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)[::-1]


    percentile_99 = np.percentile(k_distances, 99)
    percentile_95 = np.percentile(k_distances, 95)
    percentile_90 = np.percentile(k_distances, 90)


    y_max = min(percentile_95 * 1.5, percentile_99)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


    ax1.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1)
    ax1.set_xlabel('Data Points (sorted by K-distance, descending)', fontsize=12)
    ax1.set_ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
    ax1.set_title(f'K-distance Graph - Full View (K={k})', fontsize=14)
    ax1.grid(True, alpha=0.3)


    if Config.EPS is not None:
        ax1.axhline(y=Config.EPS, color='r', linestyle='--', linewidth=2,
                   label=f'Current EPS={Config.EPS}')
    ax1.axhline(y=percentile_95, color='orange', linestyle=':', linewidth=1.5,
               label=f'95th percentile={percentile_95:.4f}')
    ax1.legend(loc='upper right')


    ax2.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1.5)
    ax2.set_xlabel('Data Points (sorted by K-distance, descending)', fontsize=12)
    ax2.set_ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
    ax2.set_title(f'K-distance Graph - Zoomed View (Y-axis limited)\nFind the "Elbow": Where curve drops sharply then flattens',
                 fontsize=14)
    ax2.set_ylim(0, y_max)
    ax2.grid(True, alpha=0.3)


    if Config.EPS is not None:
        ax2.axhline(y=Config.EPS, color='r', linestyle='--', linewidth=2,
                   label=f'Current EPS={Config.EPS}')


    suggested_eps_low = percentile_90
    suggested_eps_high = percentile_95
    ax2.axhspan(suggested_eps_low, suggested_eps_high, alpha=0.2, color='green',
               label=f'Suggested range: [{suggested_eps_low:.4f}, {suggested_eps_high:.4f}]')

    ax2.legend(loc='upper right', fontsize=10)


    stats_text = f'Statistics:\n'
    stats_text += f'90th percentile: {percentile_90:.4f}\n'
    stats_text += f'95th percentile: {percentile_95:.4f}\n'
    stats_text += f'99th percentile: {percentile_99:.4f}\n'
    stats_text += f'Max distance: {k_distances[0]:.4f}'

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path is None:
        save_path = os.path.join(Config.PLOT_PATH, 'k_distance_graph.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"\nK-distance graph saved to: {save_path}")
    print(f"\nSuggested EPS range: [{suggested_eps_low:.4f}, {suggested_eps_high:.4f}]")
    print(f"Current EPS setting: {Config.EPS}")
    if Config.EPS < suggested_eps_low:
        print("Warning: Current EPS may be too small; clustering may be too strict.")
    elif Config.EPS > suggested_eps_high:
        print("Warning: Current EPS may be too large; distant clusters may be merged.")
    else:
        print("OK: Current EPS is within the suggested range.")
    plt.close()


def visualize_denoise(original_data, denoised_data, labels=None, save_path=None):


    Config.ensure_plot_dir()


    def sample_data(data, max_points):
        if len(data) > max_points:
            indices = np.random.choice(len(data), max_points, replace=False)
            return data[indices], indices
        return data, np.arange(len(data))

    original_sample, orig_indices = sample_data(original_data, Config.MAX_PLOT_POINTS)
    denoised_sample, denoise_indices = sample_data(denoised_data, Config.MAX_PLOT_POINTS)

    fig, axes = plt.subplots(1, 2, figsize=Config.FIGURE_SIZE)


    axes[0].scatter(original_sample[:, 0], original_sample[:, 1],
                   c='blue', s=Config.POINT_SIZE, alpha=Config.POINT_ALPHA, edgecolors='none')
    axes[0].set_xlabel('Longitude', fontsize=10)
    axes[0].set_ylabel('Latitude', fontsize=10)
    sample_info = f" (showing {len(original_sample)}/{len(original_data)})" if len(original_sample) < len(original_data) else ""
    axes[0].set_title(f'Before Denoising\nTotal: {len(original_data)} points{sample_info}', fontsize=12)
    axes[0].grid(True, alpha=0.3)


    axes[1].scatter(denoised_sample[:, 0], denoised_sample[:, 1],
                   c='green', s=Config.POINT_SIZE, alpha=Config.POINT_ALPHA, edgecolors='none')
    axes[1].set_xlabel('Longitude', fontsize=10)
    axes[1].set_ylabel('Latitude', fontsize=10)
    noise_count = len(original_data) - len(denoised_data)
    sample_info = f" (showing {len(denoised_sample)}/{len(denoised_data)})" if len(denoised_sample) < len(denoised_data) else ""
    axes[1].set_title(f'After Denoising\nRemaining: {len(denoised_data)}, Removed: {noise_count}{sample_info}',
                     fontsize=12)
    axes[1].grid(True, alpha=0.3)

    if save_path is None:
        save_path = os.path.join(Config.PLOT_PATH, 'denoise_comparison.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"Denoising comparison saved to: {save_path}")
    plt.close()


    if labels is not None:
        plot_clusters(original_data, labels)

def plot_clusters(data, labels, save_path=None):


    Config.ensure_plot_dir()


    def sample_by_label(data, labels, max_points):
        unique_labels = set(labels)
        sampled_data = []
        sampled_labels = []


        points_per_label = max_points // len(unique_labels)

        for label in unique_labels:
            mask = labels == label
            label_data = data[mask]

            if len(label_data) > points_per_label:
                indices = np.random.choice(len(label_data), points_per_label, replace=False)
                sampled_data.append(label_data[indices])
                sampled_labels.extend([label] * points_per_label)
            else:
                sampled_data.append(label_data)
                sampled_labels.extend([label] * len(label_data))

        return np.vstack(sampled_data), np.array(sampled_labels)

    if len(data) > Config.MAX_PLOT_POINTS:
        plot_data, plot_labels = sample_by_label(data, labels, Config.MAX_PLOT_POINTS)
        sample_info = f" (showing {len(plot_data)}/{len(data)})"
    else:
        plot_data, plot_labels = data, labels
        sample_info = ""

    plt.figure(figsize=(10, 8))


    unique_labels = set(plot_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:

            col = 'black'

        class_member_mask = (plot_labels == k)
        xy = plot_data[class_member_mask]

        if k == -1:
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=Config.POINT_SIZE,
                       alpha=0.3, edgecolors='none', label='Noise')
        else:
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=Config.POINT_SIZE,
                       alpha=Config.POINT_ALPHA, edgecolors='none')

    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    plt.title(f'DBSCAN Clustering Results{sample_info}\n'
             f'Clusters: {n_clusters}, Noise points: {n_noise}', fontsize=14)
    plt.grid(True, alpha=0.3)


    if -1 in unique_labels:
        plt.legend(loc='best', markerscale=3)

    if save_path is None:
        save_path = os.path.join(Config.PLOT_PATH, 'cluster_results.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"Clustering results saved to: {save_path}")
    plt.close()

def output_clustering_results(data, labels, save_path=None):


    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_PATH, 'clustering_results.txt')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        for point, label in zip(data, labels):
            f.write(f"{point[0]},{point[1]},{label}\n")

    print(f"Clustering results output to: {save_path}")

def read_data(file_path=None, startday=None, endday=None):


    if file_path is None:
        file_path = Config.DATA_PATH
    if startday is None:
        startday = Config.START_DAY
    if endday is None:
        endday = Config.END_DAY

    file_path = Path(file_path)
    data = []
    for day in range(startday, endday + 1):

        filename = file_path / f"order_2016110{day}"
        with open(filename, 'r') as file:
            for line in file:
                fields = line.strip().split(',')
                if len(fields) != 8:
                    continue
                point_id = fields[0]
                coordinates = list(map(float, fields[1:]))
                data.append((point_id, coordinates))

    res = [coords[-3:-1] for _, coords in data if len(coords) >= 3]
    return res

def denoise_data():
    data = read_data()
    denoised_data, _, _ = dbscan_denoise(data)
    return denoised_data

if __name__ == "__main__":
    print("=" * 60)
    print("Start DBSCAN Denoising Process")
    print("=" * 60)


    print(f"\n1. Reading data (Date range: {Config.START_DAY} - {Config.END_DAY})")
    data = read_data()
    print(f"Data points loaded: {len(data)}")
    print("Data sample:", data[:5])


    print(f"\n2. Plotting K-distance graph (K={Config.K_NEIGHBORS})")
    plot_k_distance(data)


    print(f"\n3. Executing DBSCAN denoising (eps={Config.EPS}, min_samples={Config.MIN_SAMPLES})")
    denoised_data, original_data, labels = dbscan_denoise(data)


    print("\n4. Generating visualization charts")
    visualize_denoise(original_data, denoised_data, labels)

    print("\n" + "=" * 60)
    print("Processing completed! All charts saved to plot folder")
    print("=" * 60)
