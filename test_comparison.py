"""
Comparación de rendimiento: 4 ejecuciones idénticas
Ejecuta MUTAG/GraphHD/eigenvector 4 veces para comparar tiempos.
"""

import time
import os
from concurrent.futures import ProcessPoolExecutor
from src.dataset_loader import load_tudataset
from src.encoders import GraphHDEncoder
from src.experiments import run_experiment

# Configuración
DATASET = 'MUTAG'
ENCODER = 'GraphHD'
METRIC = 'eigenvector'
DIM = 10000
N_REP = 100
DATA_DIR = './data'

def run_job(args):
    """Función para el worker"""
    dataset, metric, dim, n_rep, data_dir, job_id = args
    from src.dataset_loader import load_tudataset
    from src.encoders import GraphHDEncoder
    from src.experiments import run_experiment
    
    start_job = time.time()
    graphs, labels = load_tudataset(data_dir, dataset)
    encoder = GraphHDEncoder(dim=dim)
    result = run_experiment(graphs, labels, encoder, metric, n_repetitions=n_rep)
    elapsed = time.time() - start_job
    print(f"  > Job {job_id} finalizado en {elapsed:.2f}s")
    return result

def test_sequential():
    print(f"\n{'='*60}")
    print("MODO SECUENCIAL (4 ejecuciones idénticas)")
    print(f"{'='*60}")
    
    start = time.time()
    for i in range(4):
        print(f"Ejecución {i+1}/4...")
        run_job((DATASET, METRIC, DIM, N_REP, DATA_DIR, i+1))
    
    elapsed = time.time() - start
    print(f"Tiempo total secuencial: {elapsed:.2f}s")
    return elapsed

def test_parallel(n_workers=4):
    print(f"\n{'='*60}")
    print(f"MODO PARALELO ({n_workers} workers, 4 ejecuciones idénticas)")
    print(f"{'='*60}")
    
    start = time.time()
    
    jobs = [(DATASET, METRIC, DIM, N_REP, DATA_DIR, i+1) for i in range(4)]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(run_job, jobs))
    
    elapsed = time.time() - start
    print(f"Tiempo total paralelo: {elapsed:.2f}s")
    return elapsed

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    print(f"Comparación: 4x {DATASET}/{ENCODER}/{METRIC}")
    
    # 1. Ejecutar paralelo primero (para que el usuario vea el resultado rápido)
    time_par = test_parallel(n_workers=4)
    
    # 2. Ejecutar secuencial
    time_seq = test_sequential()
    
    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE GANANCIA")
    print(f"{'='*60}")
    print(f"Tiempo secuencial (4 runs): {time_seq:.2f}s")
    print(f"Tiempo paralelo (4 runs):   {time_par:.2f}s")
    speedup = time_seq / time_par
    print(f"Speedup real: {speedup:.2f}x")
    print(f"Ahorro de tiempo: {time_seq - time_par:.2f}s")
