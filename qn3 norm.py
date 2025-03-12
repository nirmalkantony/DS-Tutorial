import numpy as np

# Define data points (2D features)
X_class1 = np.array([[4, 3], [6, 3]])  # Class 1
X_class2 = np.array([[2, 1], [3, 2]])  # Class 2

# Compute mean vectors
mean1 = np.mean(X_class1, axis=0)
mean2 = np.mean(X_class2, axis=0)

# Compute within-class scatter matrix Sw
S1 = np.dot((X_class1 - mean1).T, (X_class1 - mean1))
S2 = np.dot((X_class2 - mean2).T, (X_class2 - mean2))
Sw = S1 + S2

# Compute between-class scatter matrix Sb
mean_diff = (mean1 - mean2).reshape(2, 1)
Sb = np.dot(mean_diff, mean_diff.T)

# Compute eigenvalues and eigenvectors of Sw^(-1) * Sb
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

# Select the eigenvector corresponding to the largest eigenvalue
lda_vector = eigvecs[:, np.argmax(eigvals)]

print("LDA Projection Vector:\n", lda_vector)
