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
            'avg_time_per_rep': result['total_time_sec'] / n_rep,
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

def print_summary_table(output_file):
    """Lee el CSV y muestra una tabla bonita por consola."""
    if not os.path.exists(output_file):
        return
    
    try:
        with csv_lock:
            df = pd.read_csv(output_file)
        
        if df.empty:
            return
            
        print("\n" + "="*85)
        print(f" {'REPORTE DE PROGRESO':^83}")
        print("="*85)
        
        # Formatear columnas para visualización
        report_df = df.copy()
        report_df['F1 Score (Mean ± SD)'] = report_df.apply(
            lambda x: f"{x['f1_mean']:.4f} ± {x['f1_std']:.4f}", axis=1
        )
        report_df['Avg Time/Run'] = report_df['avg_time_per_rep'].apply(lambda x: f"{x:.2f}s")
        
        # Seleccionar y mostrar
        display_cols = ['Dataset', 'Encoder', 'Centrality', 'F1 Score (Mean ± SD)', 'Avg Time/Run']
        print(report_df[display_cols].to_string(index=False))
        print("="*85 + "\n")
        
    except Exception as e:
        pass # Silencioso para no romper el flujo principal

def main():
    parser = argparse.ArgumentParser(description='Benchmark Paralelo GraphHD')
    parser.add_argument('--output', type=str, default='benchmark_parallel_results.csv', 
                        help='Archivo CSV de salida')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Directorio de datos')
    parser.add_argument('--dim', type=int, default=10000, 
                        help='Dimensión de hipervectores')
    parser.add_argument('--n_rep', type=int, default=100, 
                        help='Número de repeticiones por experimento')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Número de workers (default: CPU count)')
    args = parser.parse_args()

    # Configuración
    datasets = ['MUTAG', 'PROTEINS', 'ENZYMES', 'NCI1', 'DD', 'PTC_FM']
    encoders = ['GraphHD', 'GraphOrder', 'GraphHDLevelPerm']
    metrics = ['pagerank', 'degree', 'closeness', 'betweenness', 'eigenvector', 'katz']
    
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
                    # Mostrar tabla actualizada
                    print_summary_table(args.output)
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
