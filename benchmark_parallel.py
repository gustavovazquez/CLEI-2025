"""
Benchmark Paralelo para GraphHD
Ejecuta experimentos en paralelo usando ProcessPoolExecutor.
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
    from src.encoders import GraphHDEncoder, GraphHDOrderEncoder, GraphHDLevelEncoder
    from src.experiments import run_experiment
    
    encoder_classes = {
        'GraphHD': GraphHDEncoder,
        'GraphOrder': GraphHDOrderEncoder,
        'GraphHDLevel': GraphHDLevelEncoder
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

def load_existing_results(output_file):
    """Carga resultados existentes para resumibilidad."""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            completed = set(zip(df['Dataset'], df['Encoder'], df['Centrality']))
            print(f"Cargados {len(completed)} resultados existentes de {output_file}")
            return completed
        except Exception as e:
            print(f"Error leyendo {output_file}: {e}")
    return set()

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
    parser = argparse.ArgumentParser(description='Benchmark Paralelo GraphHD')
    parser.add_argument('--output', type=str, default='benchmark_parallel_results.csv', 
                        help='Archivo CSV de salida')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Directorio de datos')
    parser.add_argument('--dim', type=int, default=10000, 
                        help='Dimensión de hipervectores')
    parser.add_argument('--n_rep', type=int, default=1, 
                        help='Número de repeticiones por experimento')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Número de workers (default: CPU count)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset(s) a ejecutar (default: todos). Ej: --datasets MUTAG PROTEINS')
    parser.add_argument('--encoders', type=str, nargs='+', default=None,
                        help='Encoder(s) a ejecutar (default: todos). Opciones: GraphHD GraphOrder GraphHDLevel')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help='Métrica(s) de centralidad (default: todas). Opciones: pagerank degree closeness betweenness eigenvector katz')
    args = parser.parse_args()

    # Configuración - valores por defecto
    all_datasets = ['MUTAG', 'PROTEINS', 'ENZYMES', 'NCI1', 'DD', 'PTC_FM']
    all_encoders = ['GraphHD', 'GraphOrder', 'GraphHDLevel']
    all_metrics = ['pagerank', 'degree', 'closeness', 'betweenness', 'eigenvector', 'katz']
    
    # Usar valores especificados o todos por defecto
    datasets = args.datasets if args.datasets else all_datasets
    encoders = args.encoders if args.encoders else all_encoders
    metrics = args.metrics if args.metrics else all_metrics
    
    # Determinar número de workers
    n_workers = args.workers or multiprocessing.cpu_count()
    print(f"Usando {n_workers} workers paralelos")
    
    # Cargar resultados existentes para resumibilidad
    completed = load_existing_results(args.output)
    
    # Generar todos los jobs pendientes
    all_jobs = list(product(datasets, encoders, metrics))
    pending_jobs = [
        (ds, enc, met, args.dim, args.n_rep, args.data_dir)
        for ds, enc, met in all_jobs
        if (ds, enc, met) not in completed
    ]
    
    print(f"Total de experimentos: {len(all_jobs)}")
    print(f"Experimentos completados: {len(completed)}")
    print(f"Experimentos pendientes: {len(pending_jobs)}")
    
    if not pending_jobs:
        print("¡Todos los experimentos ya están completados!")
        return
    
    # Ejecutar en paralelo
    start_time = time.time()
    completed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Enviar todos los jobs
        future_to_job = {
            executor.submit(run_single_job, job): job 
            for job in pending_jobs
        }
        
        # Procesar resultados a medida que completan
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            ds, enc, met = job[:3]
            
            try:
                result = future.result()
                
                if result['status'] == 'success':
                    save_result(result, args.output)
                    completed_count += 1
                    print(f"✓ [{completed_count}/{len(pending_jobs)}] "
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
    print(f"Exitosos: {completed_count}, Errores: {error_count}")
    print(f"Resultados guardados en: {args.output}")

if __name__ == "__main__":
    # Necesario para Windows
    multiprocessing.freeze_support()
    main()
