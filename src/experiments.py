import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
<<<<<<< HEAD
from .hdc_utils import bundle
from .models import BinaryHypervector, FloatHypervector, BipolarHypervector
=======
from .hdc_utils import cosine_similarity, sign, bundle
>>>>>>> parent of 344d83b (scripts finales)
import time


class HDCClassifier:
    def __init__(self):
        self.class_hvs = {}

    def train(self, X_train, y_train):
        """Creates class hypervectors by bundling graph vectors of each class."""
        self.class_hvs = {}
        unique_labels = np.unique(y_train)

        for label in unique_labels:
            class_vectors = [X_train[i] for i, l in enumerate(y_train) if l == label]
            if class_vectors:
<<<<<<< HEAD
                self.class_hvs[label] = bundle(class_vectors)

    def predict(self, X_test):
        """
        Batch prediction using GPU matrix operations.
        Computes all similarities in a single matmul instead of nested loops.
        """
        if not X_test or not self.class_hvs:
            return np.array([])

        labels = list(self.class_hvs.keys())

        # Stack all vectors as GPU tensors
        X_matrix = torch.stack([x.data for x in X_test])       # (N, D)
        C_matrix = torch.stack([self.class_hvs[l].data for l in labels])  # (C, D)

        sample = X_test[0]

        if isinstance(sample, BinaryHypervector):
            # Efficient Hamming via matmul trick:
            # XOR_count(a,b) = sum(a) + sum(b) - 2*dot(a,b)
            # Hamming_sim = 1 - XOR_count / D
            X_float = X_matrix.float()
            C_float = C_matrix.float()
            dot_products = X_float @ C_float.T           # (N, C)
            X_sums = X_float.sum(dim=1, keepdim=True)    # (N, 1)
            C_sums = C_float.sum(dim=1, keepdim=True).T  # (1, C)
            xor_counts = X_sums + C_sums - 2 * dot_products
            similarities = 1.0 - xor_counts / sample.dim

        elif isinstance(sample, BipolarHypervector):
            # Bipolar cosine: dot(a,b) / D
            similarities = (X_matrix.float() @ C_matrix.float().T) / sample.dim

        else:  # FloatHypervector
            # Cosine similarity via normalized matmul
            X_norm = X_matrix / X_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
            C_norm = C_matrix / C_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
            similarities = X_norm @ C_norm.T

        best_indices = similarities.argmax(dim=1).cpu().numpy()
        return np.array([labels[i] for i in best_indices])

=======
                bundled = bundle(class_vectors)
                self.class_hvs[label] = sign(bundled)

    def predict(self, X_test):
        """Predicts class labels for test graph vectors."""
        predictions = []
        for x in X_test:
            best_label = None
            max_sim = -1.1
            
            for label, class_hv in self.class_hvs.items():
                sim = cosine_similarity(x, class_hv)
                if sim > max_sim:
                    max_sim = sim
                    best_label = label
            predictions.append(best_label)
        return np.array(predictions)
>>>>>>> parent of 344d83b (scripts finales)

def run_experiment(graphs, labels, encoder, centrality_metric, n_repetitions=10):
    """Runs the full experiment pipeline n times and measures total time."""
    accuracies = []
    f1_scores_list = []

    start_time = time.time()
    for i in range(n_repetitions):
        # 1. Prepare library (randomized per repetition)
        encoder.prepare_library(graphs)

        # 2. Encode graphs
        X = [encoder.encode(g, centrality_metric=centrality_metric) for g in graphs]
        y = np.array(labels)

        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=None
        )

        # 4. Train and Predict (batch GPU)
        clf = HDCClassifier()
        clf.train(X_train, y_train)
        y_pred = clf.predict(X_test)

        # 5. Metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores_list.append(f1_score(y_test, y_pred, average='weighted'))

    end_time = time.time()
    total_time = end_time - start_time

    return {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores_list),
        'f1_std': np.std(f1_scores_list),
        'total_time_sec': total_time
    }
