import os
import argparse
from src.dataset_loader import load_tudataset
from src.encoders import GraphHDEncoder, GraphOrderEncoder, GraphHDLevelEncoder, GraphOrderLevelEncoder
from src.experiments import run_experiment
from src.visualization import generate_results_table, plot_comparison, plot_times

def main():
    parser = argparse.ArgumentParser(description='GraphHD and GraphOrder Experiment Runner')
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name (e.g., MUTAG, PROTEINS)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing datasets')
    parser.add_argument('--dim', type=int, default=10000, help='Hypervector dimension')
    parser.add_argument('--n_rep', type=int, default=10, help='Number of repetitions')
    parser.add_argument('--metrics', nargs='+', default=['pagerank', 'degree', 'closeness'], 
                        help='Centrality metrics to test')
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}...")
    try:
        graphs, labels = load_tudataset(args.data_dir, args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Loaded {len(graphs)} graphs.")
    
    encoders = {
        'GraphHD': GraphHDEncoder(dim=args.dim),
        'GraphOrder': GraphOrderEncoder(dim=args.dim),
        'GraphHDLevel': GraphHDLevelEncoder(dim=args.dim),
        'GraphOrderLevel': GraphOrderLevelEncoder(dim=args.dim)
    }
    
    all_results = []
    
    for enc_name, encoder in encoders.items():
        for metric in args.metrics:
            print(f"Running Experiment: {enc_name} with {metric}...")
            res = run_experiment(graphs, labels, encoder, metric, n_repetitions=args.n_rep)
            
            # Record result
            res.update({
                'Dataset': args.dataset,
                'Encoder': enc_name,
                'Centrality': metric
            })
            all_results.append(res)
            print(f"  F1: {res['f1_mean']:.4f} \u00b1 {res['f1_std']:.4f} | Time: {res['total_time_sec']:.2f}s")

    # 1. Generate Table
    df = generate_results_table(all_results)
    print("\nGlobal Results Table:")
    print(df[['Encoder', 'Centrality', 'f1_mean', 'total_time_sec']].to_string())
    
    # Save to CSV
    csv_path = f"results_{args.dataset}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # 2. Generate Plots
    f1_plot_path = f"f1_comparison_{args.dataset}.png"
    plot_comparison(df, metric='f1_mean', output_path=f1_plot_path)
    
    time_plot_path = f"time_comparison_{args.dataset}.png"
    plot_times(df, output_path=time_plot_path)
    
    print(f"Plots saved to {f1_plot_path} and {time_plot_path}")

if __name__ == "__main__":
    main()
