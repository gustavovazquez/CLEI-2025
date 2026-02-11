from src.dataset_loader import load_tudataset
from src.encoders import GraphHDEncoder
from src.experiments import run_experiment
import time

graphs, labels = load_tudataset('./data', 'MUTAG')
print(f'Loaded {len(graphs)} graphs')

enc = GraphHDEncoder(dim=10000)
t0 = time.time()
result = run_experiment(graphs, labels, enc, 'degree', n_repetitions=10)
t1 = time.time()

print(f"F1: {result['f1_mean']:.4f} +/- {result['f1_std']:.4f}")
print(f"Time: {t1-t0:.2f}s")
