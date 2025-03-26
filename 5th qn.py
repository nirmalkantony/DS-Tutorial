import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CustomSVM:
    def __init__(self, lr=0.001, reg=0.01, epochs=1000):
        """
        Custom Support Vector Machine (SVM) classifier using stochastic gradient descent.
        """
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Train the SVM model using gradient descent.
        """
        samples, features = X.shape
        y_transformed = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(features)
        self.bias = 0
        
        for _ in range(self.epochs):
            for index, x_i in enumerate(X):
                condition = y_transformed[index] * (np.dot(x_i, self.weights) - self.bias) >= 1
                
                if condition:
                    self.weights -= self.lr * (2 * self.reg * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.reg * self.weights - np.dot(x_i, y_transformed[index]))
                    self.bias -= self.lr * y_transformed[index]
    
    def predict(self, X):
        """
        Predict class labels for given input data.
        """
        return np.sign(np.dot(X, self.weights) - self.bias)
    
    def decision_boundary(self, X):
        """
        Compute the decision boundary.
        """
        return np.dot(X, self.weights) - self.bias

# Function to plot decision boundary
def visualize_boundary(svm, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm.decision_boundary(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.show()

# Example usage
if __name__ == "__main__":
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=1, 
                               n_clusters_per_class=1)
    y = np.where(y == 0, -1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CustomSVM(lr=0.001, reg=0.01, epochs=1000)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"SVM Model Accuracy: {accuracy:.4f}")
    
    visualize_boundary(model, X_train, y_train)
