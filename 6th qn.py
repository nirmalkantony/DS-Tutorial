import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

class HierarchicalClusterModel:
    def __init__(self, num_clusters=2, linkage_method='complete'):
        """
        Custom implementation of hierarchical clustering algorithm.
        """
        self.num_clusters = num_clusters
        self.linkage_method = linkage_method
        self.labels = None
        self.link_matrix = None
    
    def train(self, dataset):
        """
        Perform hierarchical clustering and compute the linkage matrix.
        """
        dist_matrix = pdist(dataset)
        self.link_matrix = linkage(dist_matrix, method=self.linkage_method)
        
        from scipy.cluster.hierarchy import fcluster
        self.labels = fcluster(self.link_matrix, t=self.num_clusters, criterion='maxclust') - 1
    
    def display_dendrogram(self):
        """
        Visualize the dendrogram for hierarchical clustering.
        """
        if self.link_matrix is None:
            raise ValueError("Clustering model has not been trained yet.")
        
        plt.figure(figsize=(10, 6))
        dendrogram(self.link_matrix)
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage_method.capitalize()} Linkage)')
        plt.xlabel('Data Point Index')
        plt.ylabel('Distance')
        plt.show()

# Generate sample dataset
np.random.seed(42)
data_points = np.concatenate([
    np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[0, 5], scale=0.5, size=(50, 2))
])

# Apply hierarchical clustering
clustering_model = HierarchicalClusterModel(num_clusters=3, linkage_method='complete')
clustering_model.train(data_points)

# Plot clustering results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(data_points[:, 0], data_points[:, 1], c=clustering_model.labels, cmap='viridis')
plt.title('Clustering Visualization')

# Generate dendrogram
plt.subplot(1, 2, 2)
clustering_model.display_dendrogram()
plt.tight_layout()
plt.show()
