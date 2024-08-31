import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_neighbors = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_neighbors]
            predictions.append(np.argmax(np.bincount(nearest_labels)))
        return np.array(predictions)

    def plot_classification(self, X_train, y_train, X_test, y_test, ax1):
        ax1.clear()
        ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data')
        ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', edgecolor='k', 
                    label='Test Data', linewidths=2)
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title('KNN Classification')
        ax1.legend()

    def animate_classification(self, X_train, y_train, X_test, y_test):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot all data on ax2 once with 'o' markers
        self.ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o')
        self.ax2.set_xlabel('Feature 1')
        self.ax2.set_ylabel('Feature 2')
        self.ax2.set_title('Original Data')
        self.ax2.legend()

        def update(frame):
            if frame < len(X_test):
                x = X_test[frame]
                distances = np.linalg.norm(self.X_train - x, axis=1)
                nearest_neighbors = np.argsort(distances)[:self.n_neighbors]
                nearest_labels = self.y_train[nearest_neighbors]
                prediction = np.argmax(np.bincount(nearest_labels))
                self.plot_classification(X_train, y_train, X_test[:frame+1], y_test[:frame+1], self.ax1)
                self.ax1.scatter(x[0], x[1], c='red', marker='o', edgecolor='k', s=100, label='Current Test Point')
                self.ax1.legend()

        anim = FuncAnimation(self.fig, update, frames=len(X_test), repeat=False)
        plt.show()

# Example usage
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features for simplicity
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNN(n_neighbors=7)
knn.fit(X_train, y_train)
knn.animate_classification(X_train, y_train, X_test, y_test)
