"""
Benchmark Paralelo específico para NCI1, PTC_FM y DD.
Ejecuta experimentos en paralelo usando 4 workers por defecto.
Cada worker procesa una combinación (Dataset, Encoder, Métrica).
"""

import os
import time
import pandas as pd
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing
import threading

# Lock global para escritura thread-safe al CSV
csv_lock = threading.Lock()

def run_single_job(job_args):
    """
    Ejecuta UN experimento completo (dataset + encoder + métrica).
    Esta función corre en un proceso separado.
    """
    dataset_name, encoder_name, metric, dim, n_rep, data_dir = job_args
    
    # Importar aquí para evitar problemas de pickling en multiprocessing
    from src.dataset_loader import load_tudataset
    from src.encoders import GraphHDEncoder, GraphOrderEncoder, GraphHDLevelPermEncoder
    from src.experiments import run_experiment
    
    encoder_classes = {
        'GraphHD': GraphHDEncoder,
        'GraphOrder': GraphOrderEncoder,
        'GraphHDLevelPerm': GraphHDLevelPermEncoder
    }
    
    try:
        # Cargar dataset
        graphs, labels = load_tudataset(data_dir, dataset_name)
        
        # Crear encoder
        encoder_class = encoder_classes[encoder_name]
        encoder = encoder_class(dim=dim)
        
        # Ejecutar experimento (incluye las n_rep repeticiones)
        result = run_experiment(graphs, labels, encoder, metric, n_repetitions=n_rep)
        
        return {
            'Dataset': dataset_name,
            'Encoder': encoder_name,
            'Centrality': metric,
            'f1_mean': result['f1_mean'],
            'f1_std': result['f1_std'],
            'accuracy_mean': result['accuracy_mean'],
            'accuracy_std': result['accuracy_std'],
            'total_time_sec': result['total_time_sec'],
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'Dataset': dataset_name,
            'Encoder': encoder_name,
            'Centrality': metric,
            'status': 'error',
            'error_msg': str(e)
        }

def save_result(result, output_file):
    """Guarda un resultado al CSV de forma atómica."""
    if result['status'] != 'success':
        return
    
    row = {k: v for k, v in result.items() if k not in ['status', 'error_msg']}
    df = pd.DataFrame([row])
    
    # Escribir con lock para thread-safety
    with csv_lock:
        header = not os.path.exists(output_file)
        df.to_csv(output_file, mode='a', header=header, index=False)

def main():
    parser = argparse.ArgumentParser(description='Benchmark Paralelo Específico (NCI1, PTC_FM, DD)')
    parser.add_argument('--output', type=str, default='results_specific.csv', 
                        help='Archivo CSV de salida')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Directorio de datos')
    parser.add_argument('--dim', type=int, default=10000, 
                        help='Dimensión de hipervectores')
    parser.add_argument('--n_rep', type=int, default=100, 
                        help='Número de repeticiones por experimento')
    parser.add_argument('--workers', type=int, default=16, 
                        help='Número de workers (default: 4)')
    args = parser.parse_args()

    # Configuración de base de datos solicitada
    datasets = ['NCI1', 'PTC_FM', 'DD']
    encoders = ['GraphHD', 'GraphOrder', 'GraphHDLevelPerm']
    # Reutilizamos métricas comunes o puedes ajustar
    metrics = ['eigenvector', 'degree', 'pagerank']
    
    print(f"Usando {args.workers} workers paralelos")
    print(f"Datasets: {datasets}")
    print(f"Repeticiones: {args.n_rep}")
    
    all_jobs = list(product(datasets, encoders, metrics))
    
    start_time = time.time()
    completed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(run_single_job, (ds, enc, met, args.dim, args.n_rep, args.data_dir)): (ds, enc, met) 
            for ds, enc, met in all_jobs
        }
        
        for future in as_completed(future_to_job):
            ds, enc, met = future_to_job[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    save_result(result, args.output)
                    completed_count += 1
                    print(f"✓ [{completed_count}/{len(all_jobs)}] "
                          f"{ds}/{enc}/{met}: F1={result['f1_mean']:.4f} "
                          f"({result['total_time_sec']:.1f}s)")
                else:
                    error_count += 1
                    print(f"✗ {ds}/{enc}/{met}: {result.get('error_msg', 'Unknown error')}")
            except Exception as e:
                error_count += 1
                print(f"✗ {ds}/{enc}/{met}: Exception - {e}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Benchmark completado en {elapsed:.1f} segundos")
    print(f"Resultados guardados en: {args.output}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
