import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class CustomRandomForest:
    def __init__(self, n_trees=100, max_depth=None, max_features='sqrt', random_state=None):
        """
        Initialize the Custom Random Forest model.
        
        Parameters:
        - n_trees: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        - max_features: Number of features considered at each split ('sqrt' for square root of total features)
        - random_state: Random seed for reproducibility
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []  # Stores trained decision trees
        self.feature_indices = []  # Stores selected feature indices per tree
    
    def fit(self, X, y):
        """
        Train the Custom Random Forest using bootstrapped samples.
        
        Parameters:
        - X: Feature matrix (num_samples, num_features)
        - y: Target labels (num_samples,)
        """
        np.random.seed(self.random_state)
        num_samples, num_features = X.shape
        
        # Determine the number of features to use per tree
        if self.max_features == 'sqrt':
            num_selected_features = int(np.sqrt(num_features))
        elif isinstance(self.max_features, int):
            num_selected_features = self.max_features
        else:
            num_selected_features = num_features
        
        self.trees = []
        self.feature_indices = []
        
        for _ in range(self.n_trees):
            # Generate a bootstrap sample
            X_sample, y_sample = resample(X, y)
            
            # Randomly choose feature subset
            selected_features = np.random.choice(num_features, num_selected_features, replace=False)
            X_sample = X_sample[:, selected_features]
            
            # Train individual decision tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            
            # Store trained tree and selected feature indices
            self.trees.append(tree)
            self.feature_indices.append(selected_features)
    
    def predict_proba(self, X):
        """
        Compute class probabilities by averaging predictions from all trees.
        
        Parameters:
        - X: Feature matrix (num_samples, num_features)
        
        Returns:
        - avg_probabilities: Array (num_samples, num_classes) with class probabilities
        """
        num_samples = X.shape[0]
        probabilities = []
        
        for tree, features in zip(self.trees, self.feature_indices):
            X_subset = X[:, features]
            probabilities.append(tree.predict_proba(X_subset))
        
        # Average the probabilities from all trees
        avg_probabilities = np.mean(probabilities, axis=0)
        return avg_probabilities
    
    def predict(self, X):
        """
        Predict class labels using majority voting from all trees.
        
        Parameters:
        - X: Feature matrix (num_samples, num_features)
        
        Returns:
        - predicted_labels: Array (num_samples,) with class predictions
        """
        avg_probabilities = self.predict_proba(X)
        return np.argmax(avg_probabilities, axis=1)

# Example usage
if __name__ == "__main__":
    # Load sample dataset
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Custom Random Forest
    model = CustomRandomForest(n_trees=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate model accuracy
    model_accuracy = accuracy_score(y_test, predictions)
    print(f"Custom Random Forest Accuracy: {model_accuracy:.4f}")
