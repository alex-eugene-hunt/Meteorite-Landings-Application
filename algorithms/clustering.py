import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle

def generate_elbow_report():
    """
    Generate an elbow report to find the optimal K for clustering.
    """
    # Load dataset
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["reclat", "reclong", "mass (g)"])  # Clean data

    if df.empty:
        print("Dataset is empty after cleaning.")
        return

    X = df[["reclat", "reclong", "mass (g)"]].to_numpy()

    # Check if dataset is large enough for clustering
    if X.shape[0] < 2:
        print("Not enough data points for clustering.")
        return

    # Elbow method: compute inertia for different values of K
    inertias = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeansScratch(n_clusters=k, max_iter=100, tol=1e-4)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot the elbow graph
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_values, inertias, marker="o", linestyle="--", color="b")
    ax.set_title("Elbow Method for Optimal K", fontsize=10)
    ax.set_xlabel("Number of Clusters (K)", fontsize=8)
    ax.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=8)
    ax.grid(True)

    return fig


class KMeansScratch:
    def __init__(self, n_clusters=5, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.inertia_ = None
        self.labels = None

    def fit(self, X):
        """
        Fit the KMeans model to the dataset.
        """
        np.random.seed(42)
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # Assign labels based on closest centroid
            distances = self._calculate_distances(X)
            new_labels = np.argmin(distances, axis=1)

            # Calculate new centroids
            new_centroids = np.array([X[new_labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) <= self.tol):
                break

            self.centroids = new_centroids
            self.labels = new_labels

        self.labels = new_labels  # Final labels

        # Calculate inertia (sum of squared distances to the nearest centroid)
        self.inertia_ = np.sum((X - self.centroids[self.labels]) ** 2)

    def predict(self, X):
        """
        Predict the cluster for each sample in X.
        """
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)

    def _calculate_distances(self, X):
        """
        Calculate the distances from each point to each centroid.
        """
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def save_model(self, file_path):
        """
        Save the KMeans model to a file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        """
        Load a KMeans model from a file.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)


def run(k=5):
    """
    Run the clustering algorithm with a user-specified number of clusters (k).

    Args:
        k (int): Number of clusters to generate.

    Returns:
        List of Matplotlib figures for visualization,
        cluster centers,
        updated DataFrame with cluster assignments.
    """
    # Load dataset
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["reclat", "reclong", "mass (g)", "year"])  # Clean data

    # Features for clustering
    X = df[["reclat", "reclong", "mass (g)"]].to_numpy()

    # Custom KMeans clustering from scratch
    def kmeans_scratch(X, n_clusters, max_iter=100, tol=1e-4):
        np.random.seed(42)
        centers = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        for _ in range(max_iter):
            distances = np.linalg.norm(X[:, None] - centers, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
            if np.linalg.norm(new_centers - centers) < tol:
                break
            centers = new_centers
        return labels, centers

    labels, centers = kmeans_scratch(X, n_clusters=k)

    # Assign cluster labels to the DataFrame
    df["cluster"] = labels

    # Create the three plots
    plots = []

    # 1. Clustered Meteorites on a World Map
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    from mpl_toolkits.basemap import Basemap
    m = Basemap(projection="mill", llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution="c", ax=ax1)
    m.drawcoastlines()
    m.drawcountries()

    # Plot individual data points as black dots
    x, y = m(df["reclong"].to_numpy(), df["reclat"].to_numpy())
    m.scatter(x, y, c="black", s=10, edgecolors="none", zorder=5, alpha=0.8)

    # Draw cluster regions as semi-transparent circles
    for cluster_idx, center in enumerate(centers):
        center_lat, center_long, _ = center
        center_x, center_y = m(center_long, center_lat)

        # Calculate approximate cluster radius
        cluster_points = df[df["cluster"] == cluster_idx]
        cluster_x, cluster_y = m(cluster_points["reclong"].to_numpy(), cluster_points["reclat"].to_numpy())
        cluster_radius = max(
            np.sqrt((cluster_x - center_x) ** 2 + (cluster_y - center_y) ** 2)
        )

        # Draw semi-transparent circle for the cluster
        circle = plt.Circle(
            (center_x, center_y),
            cluster_radius * 0.3,  # Adjust scaling for better visualization
            color=f"C{cluster_idx}",  # Use a distinct color for each cluster
            alpha=0.3,
            zorder=4,
            transform=ax1.transData
        )
        ax1.add_patch(circle)

    # Add title and legend
    legend_elements = [plt.Line2D([0], [0], marker="o", color=f"C{i}", label=f"Cluster {i + 1}",
                                  markersize=10, linestyle="None", alpha=0.5) for i in range(k)]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8, title="Clusters")
    plt.title("Clustered Meteorites on World Map")
    plots.append(fig1)

    # 2. Mass Distribution Across Clusters
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    df.boxplot(column="mass (g)", by="cluster", ax=ax2, grid=False, patch_artist=True, boxprops=dict(facecolor="lightblue", color="blue"))
    ax2.set_title("Mass Distribution Across Clusters")
    ax2.set_xlabel("Cluster")
    ax2.set_xticklabels([f"{int(tick)}" for tick in ax2.get_xticks()])
    ax2.set_ylabel("Mass (g)")
    plt.suptitle("")  # Remove default title
    plots.append(fig2)

    # 3. Statistics and Averages for Each Cluster
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    # Calculate statistics for each cluster
    cluster_stats = df.groupby("cluster").agg({
        "mass (g)": ["mean", "median", "std"],
        "year": ["mean", "median", "std"]
    }).reset_index()

    # Flatten MultiIndex for easier plotting
    cluster_stats.columns = ["Cluster", "Mass Mean", "Mass Median", "Mass Std", "Year Mean", "Year Median", "Year Std"]

    # Plot the statistics as grouped bars
    x = np.arange(len(cluster_stats["Cluster"]))  # Cluster indices
    width = 0.2  # Bar width
    ax3.bar(x - width, cluster_stats["Mass Mean"], width, label="Mass Mean", color="blue", alpha=0.7)
    ax3.bar(x, cluster_stats["Mass Median"], width, label="Mass Median", color="green", alpha=0.7)
    ax3.bar(x + width, cluster_stats["Mass Std"], width, label="Mass Std", color="orange", alpha=0.7)

    # Add labels and title
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{int(tick) + 1}" for tick in cluster_stats["Cluster"]])
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Mass (g)")
    ax3.set_title("Cluster Statistics (Mass Mean, Median, Std)")
    ax3.legend()
    plots.append(fig3)

    return plots, centers, df  # Return all required values

# If this script is run directly, execute the clustering
if __name__ == "__main__":
    run()
