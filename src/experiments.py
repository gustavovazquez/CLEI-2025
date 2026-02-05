import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from .hdc_utils import cosine_similarity, sign, bundle

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

def run_experiment(graphs, labels, encoder, centrality_metric, n_repetitions=10):
    """Runs the full experiment pipeline n times."""
    accuracies = []
    f1_scores = []
    
    for i in range(n_repetitions):
        # 1. Prepare library (randomized per repetition)
        encoder.prepare_library(graphs)
        
        # 2. Encode graphs
        X = [encoder.encode(g, centrality_metric=centrality_metric) for g in graphs]
        y = np.array(labels)
        
        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        
        # 4. Train and Predict
        clf = HDCClassifier()
        clf.train(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # 5. Metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        
    return {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores)
    }
