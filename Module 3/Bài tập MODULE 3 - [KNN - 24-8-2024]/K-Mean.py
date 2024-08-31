from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

iris = load_iris()
iris_data = iris.data
print(iris_data.shape)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
print(iris_df.head(5))

data = iris_data[:, :2] # sepal length, sepal width

# Create a figure with 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot initial data in the first subplot
ax1.scatter(data[:, 0], data[:, 1], c='gray', label='Initial Data')
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
ax1.set_title('Initial Data')
ax1.legend()

# Plot original data in the second subplot
scatter = ax2.scatter(iris_data[:, 0], iris_data[:, 1], c=iris.target, cmap='viridis', label='Original Data')
ax2.set_xlabel('Sepal Length')
ax2.set_ylabel('Sepal Width')
ax2.set_title('Original Data')
ax2.legend()

# Display the plot
plt.tight_layout()
plt.show()


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300):
        self.n_clusters = n_clusters # number of clusters
        self.max_iter = max_iter # number of iterations
        self.cluster_centers = None # cluster centers
        self.clusters = None # cluster labels of each data point
    
    def initialize_centroids(self, data):
        np.random.seed(42)

        # Randomly select the cluster centers
        cluster_centers = data[np.random.choice(range(data.shape[0]), self.n_clusters, replace=False)]

        self.cluster_centers = cluster_centers
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def assign_clusters(self, data, cluster_centers):
        # Compute the distances between the data point and the cluster centers
        distances = [[self.euclidean_distance(x, center) for center in cluster_centers] for x in data]
        # Assign the cluster label
        return np.argmin(distances, axis=1)

    def update_centroids(self, data, clusters):
        # Update the cluster centers
        return np.array([data[clusters == k].mean(axis=0) for k in range(self.n_clusters)])

    def plot_clusters(self, data, iteration, ax1):
        ax1.clear()
        ax1.scatter(data[:, 0], data[:, 1], c=self.clusters, cmap='viridis', label='Data Points')
        ax1.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], c='red', s=100, label='Cluster Centers')
        ax1.set_xlabel('Sepal Length')
        ax1.set_ylabel('Sepal Width')
        ax1.set_title(f'Iteration {iteration}')
        ax1.legend()

    def fit(self, data):
        self.initialize_centroids(data)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original data in the second subplot
        self.ax2.scatter(data[:, 0], data[:, 1], c=iris.target, cmap='viridis', label='Original Data')
        self.ax2.set_xlabel('Sepal Length')
        self.ax2.set_ylabel('Sepal Width')
        self.ax2.set_title('Original Data\n(the colors just mean that the data is classified into 3 classes)')
        self.ax2.legend()

        def update(frame):
            self.clusters = self.assign_clusters(data, self.cluster_centers)
            new_centroids = self.update_centroids(data, self.clusters)
            
            if np.all(self.cluster_centers == new_centroids):
                anim.event_source.stop()
            else:
                self.cluster_centers = new_centroids
                self.plot_clusters(data, frame, self.ax1)

        anim = FuncAnimation(self.fig, update, frames=self.max_iter, repeat=False)
        plt.show()

        return self.cluster_centers

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
