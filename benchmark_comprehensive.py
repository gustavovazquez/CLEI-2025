import os
import time
import pandas as pd
import numpy as np
import argparse
from src.dataset_loader import load_tudataset
from src.encoders import GraphHDEncoder, GraphOrderEncoder, GraphHDLevelEncoder, GraphOrderLevelEncoder
from src.experiments import run_experiment

def main():
    parser = argparse.ArgumentParser(description='Comprehensive GraphHD Benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', help='Output CSV file')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--dim', type=int, default=10000, help='Hypervector dimension')
    parser.add_argument('--n_rep', type=int, default=100, help='Number of repetitions')
    args = parser.parse_args()

    datasets = ['MUTAG', 'PROTEINS', 'ENZYMES', 'NCI1', 'DD', 'PTC_FM']
    metrics = ['pagerank', 'degree', 'closeness', 'betweenness', 'eigenvector', 'katz']
    
    # Initialize encoders
    encoder_configs = {
        'GraphHD': GraphHDEncoder,
        'GraphOrder': GraphOrderEncoder,
        'GraphHDLevel': GraphHDLevelEncoder,
        'GraphOrderLevel': GraphOrderLevelEncoder
    }

    # Load existing results for resumability
    if os.path.exists(args.output):
        try:
            existing_df = pd.read_csv(args.output)
            print(f"Loaded {len(existing_df)} existing results from {args.output}.")
        except Exception as e:
            print(f"Error reading {args.output}, starting fresh: {e}")
            existing_df = pd.DataFrame(columns=['Dataset', 'Encoder', 'Centrality', 'f1_mean', 'f1_std', 'accuracy_mean', 'accuracy_std', 'total_time_sec'])
    else:
        existing_df = pd.DataFrame(columns=['Dataset', 'Encoder', 'Centrality', 'f1_mean', 'f1_std', 'accuracy_mean', 'accuracy_std', 'total_time_sec'])
        existing_df.to_csv(args.output, index=False)

    def is_already_done(dataset, encoder, metric):
        match = existing_df[(existing_df['Dataset'] == dataset) & 
                            (existing_df['Encoder'] == encoder) & 
                            (existing_df['Centrality'] == metric)]
        return len(match) > 0

    for ds_name in datasets:
        print(f"\nProcessing Dataset: {ds_name}")
        try:
            graphs, labels = load_tudataset(args.data_dir, ds_name)
        except Exception as e:
            print(f"Skipping {ds_name}: Error loading dataset: {e}")
            continue

        for enc_name, enc_class in encoder_configs.items():
            encoder = enc_class(dim=args.dim)
            
            for metric in metrics:
                if is_already_done(ds_name, enc_name, metric):
                    print(f"  Skipping {enc_name} - {metric} (Already done)")
                    continue
                
                print(f"  Running {enc_name} with {metric}...")
                
                # Special casing for betweenness on large datasets if needed
                if ds_name == 'DD' and metric == 'betweenness':
                    print("  Warning: Betweenness on DD can be extremely slow.")
                
                try:
                    res = run_experiment(graphs, labels, encoder, metric, n_repetitions=args.n_rep)
                    
                    # Store result
                    new_row = {
                        'Dataset': ds_name,
                        'Encoder': enc_name,
                        'Centrality': metric,
                        'f1_mean': res['f1_mean'],
                        'f1_std': res['f1_std'],
                        'accuracy_mean': res['accuracy_mean'],
                        'accuracy_std': res['accuracy_std'],
                        'total_time_sec': res['total_time_sec']
                    }
                    
                    # Append to file incrementally
                    pd.DataFrame([new_row]).to_csv(args.output, mode='a', header=False, index=False)
                    print(f"    F1: {res['f1_mean']:.4f} \u00b1 {res['f1_std']:.4f} in {res['total_time_sec']:.2f}s")
                    
                    # Update local df for further checking
                    existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                except Exception as e:
                    print(f"    Error running {enc_name} with {metric}: {e}")

    print(f"\nBenchmark complete. Results saved in {args.output}")

if __name__ == "__main__":
    main()
