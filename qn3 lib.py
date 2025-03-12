import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Sample dataset (2 classes, 2 features)
X = np.array([[4, 3], 
              [6, 3], 
              [2, 1], 
              [3, 2]])

# Labels (class 0 and class 1)
y = np.array([0, 0, 1, 1])

# Perform LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

print("LDA Transformed Data:\n", X_lda)
print("LDA Coefficients:\n", lda.coef_)
print("LDA Intercept:\n", lda.intercept_)
