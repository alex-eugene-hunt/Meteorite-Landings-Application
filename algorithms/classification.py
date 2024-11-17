# Script for Tab 1 (Classification)

import pandas as pd
import numpy as np
import dill as pickle  # To save the model


def calculate_feature_importance(node, feature_importance, total_samples):
    """
    Recursively calculate feature importance for a decision tree node.

    Args:
        node: Current node in the decision tree.
        feature_importance: Dictionary to store feature contributions.
        total_samples: Total number of samples in the dataset.

    Returns:
        Updated feature_importance dictionary.
    """
    if node is None or node.value is not None:  # Stop recursion at leaf nodes
        return feature_importance

    # Calculate the importance of the split at this node
    left_samples = sum(node.left.class_counts) if node.left and node.left.class_counts is not None else 0
    right_samples = sum(node.right.class_counts) if node.right and node.right.class_counts is not None else 0
    total_split_samples = left_samples + right_samples

    if total_split_samples > 0:
        # Contribution of this split to feature importance
        contribution = total_split_samples / total_samples
        feature_importance[node.feature_index] += contribution

    # Recursively calculate for left and right children
    if node.left:
        feature_importance = calculate_feature_importance(node.left, feature_importance, total_samples)
    if node.right:
        feature_importance = calculate_feature_importance(node.right, feature_importance, total_samples)

    return feature_importance


def get_model_metrics():
    """
    Compute and return evaluation metrics for the decision tree model.
    """
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["mass (g)", "year", "reclat", "reclong", "fall"])
    X = df[["mass (g)", "year", "reclat", "reclong"]].to_numpy()
    y = df["fall"].apply(lambda x: 1 if x == "Fell" else 0).to_numpy()

    # Collect mass values for Graph 1
    mass_values = df["mass (g)"].to_numpy()

    # Balanced train-test split
    def stratified_split(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        test_indices = []
        train_indices = []
        for cls, count in zip(unique_classes, class_counts):
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            split_idx = int(len(cls_indices) * test_size)
            test_indices.extend(cls_indices[:split_idx])
            train_indices.extend(cls_indices[split_idx:])
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # Train model if needed
    with open("algorithms/classification_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred, confidences = model.predict(X_test)

    # Metrics
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "mass_values": mass_values,
        "confidence_scores": confidences,
    }


# Calculate entropy for a dataset
def calculate_entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    """
    Split the dataset based on a feature and threshold.
    """
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold
    return (
        np.array(X[left_indices]),
        np.array(X[right_indices]),
        np.array(y[left_indices]),
        np.array(y[right_indices]),
    )


# Calculate information gain
def calculate_information_gain(y, left_y, right_y):
    parent_entropy = calculate_entropy(y)
    n = len(y)
    n_left, n_right = len(left_y), len(right_y)
    weighted_entropy = (n_left / n) * calculate_entropy(left_y) + (n_right / n) * calculate_entropy(right_y)
    info_gain = parent_entropy - weighted_entropy
    return info_gain


# Build a decision tree
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, class_counts=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.class_counts = class_counts


class DecisionTreeClassifier:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if depth >= self.max_depth or n_classes == 1 or n_samples < 2:
            leaf_value = np.bincount(y).argmax()
            class_counts = np.bincount(y, minlength=2)
            return DecisionTreeNode(value=leaf_value, class_counts=class_counts)

        best_gain = -1
        best_feature, best_threshold = None, None
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_X, right_X, left_y, right_y = split_dataset(X, y, feature_index, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gain = calculate_information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        if best_gain == -1:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)

        left_X, right_X, left_y, right_y = split_dataset(X, y, best_feature, best_threshold)
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        return DecisionTreeNode(
            feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child
        )

    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
        results = [self._traverse_tree(x, self.root) for x in X]
        predictions, confidences = zip(*results)
        return np.array(predictions), np.array(confidences)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            total_samples = sum(node.class_counts)
            confidence = node.class_counts[node.value] / total_samples if total_samples > 0 else 0
            return node.value, confidence
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def train_and_save_model():
    df = pd.read_csv("dataset/meteorites.csv")
    df = df.dropna(subset=["mass (g)", "year", "reclat", "reclong", "fall"])

    X = df[["mass (g)", "year", "reclat", "reclong"]].to_numpy()
    y = df["fall"].apply(lambda x: 1 if x == "Fell" else 0).to_numpy()

    def stratified_split(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        test_indices = []
        train_indices = []
        for cls, count in zip(unique_classes, class_counts):
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            split_idx = int(len(cls_indices) * test_size)
            test_indices.extend(cls_indices[:split_idx])
            train_indices.extend(cls_indices[split_idx:])
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)

    with open("algorithms/classification_model.pkl", "wb") as f:
        pickle.dump(model, f)


def predict(input_data):
    with open("algorithms/classification_model.pkl", "rb") as f:
        model = pickle.load(f)

    input_array = np.array(input_data).reshape(1, -1)
    prediction, confidence = model.predict(input_array)
    prediction = prediction[0]
    confidence = confidence[0]

    return "Fell" if prediction == 1 else "Found", confidence


if __name__ == "__main__":
    train_and_save_model()
