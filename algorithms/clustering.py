import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def run():
    # Load dataset
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["reclat", "reclong", "mass (g)", "year"])  # Clean data

    # Features for clustering
    X = df[["reclat", "reclong", "mass (g)"]]

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    # Create the three plots
    plots = []

    # 1. Clustered Meteorites on a World Map
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    m = Basemap(projection="mill", llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution="c", ax=ax1)
    m.drawcoastlines()
    m.drawcountries()
    scatter = m.scatter(df["reclong"], df["reclat"], c=df["cluster"], cmap="viridis", latlon=True, s=50, edgecolors="k", zorder=5)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Clustered Meteorites on World Map")
    plots.append(fig1)

    # 2. Mass Distribution Across Clusters
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    df.boxplot(column="mass (g)", by="cluster", ax=ax2, grid=False)
    ax2.set_title("Mass Distribution Across Clusters")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Mass (g)")
    plt.suptitle("")  # Remove default title
    plots.append(fig2)

    # 3. Year of Fall Distribution Across Clusters
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for cluster in df["cluster"].unique():
        cluster_data = df[df["cluster"] == cluster]["year"]
        ax3.hist(cluster_data, bins=15, alpha=0.6, label=f"Cluster {cluster}")
    ax3.set_title("Year of Fall Distribution Across Clusters")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    plots.append(fig3)

    # Return all plots
    return plots
