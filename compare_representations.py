import os
import pandas as pd
import numpy as np
from src.dataset_loader import load_tudataset
from src.encoders import GraphHDEncoder
from src.experiments import run_experiment
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    dataset_name = 'PROTEINS'
    dim = 10000
    n_rep = 50 # 50 repetitions for stable results
    metric = 'degree'
    
    print(f"Comparing Representations on {dataset_name} using GraphHD + {metric}")
    
    try:
        graphs, labels = load_tudataset('./data', dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    repr_types = ['binary', 'bipolar', 'float']
    results = []
    
    for r_type in repr_types:
        print(f"  Testing representation: {r_type}...")
        encoder = GraphHDEncoder(dim=dim, repr_type=r_type)
        
        res = run_experiment(graphs, labels, encoder, metric, n_repetitions=n_rep)
        
        res.update({
            'Representation': r_type,
            'Encoder': 'GraphHD',
            'Centrality': metric
        })
        results.append(res)
        print(f"    F1: {res['f1_mean']:.4f} \u00b1 {res['f1_std']:.4f} | Time: {res['total_time_sec']:.2f}s")

    df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(df[['Representation', 'f1_mean', 'f1_std', 'total_time_sec']].to_string())
    
    # Save results
    df.to_csv('repr_comparison_results.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Representation', y='f1_mean', capsize=.2)
    plt.errorbar(x=range(len(df)), y=df['f1_mean'], yerr=df['f1_std'], fmt='none', c='black')
    plt.title(f'GraphHD + {metric} Performance by Representation ({dataset_name})')
    plt.ylabel('F1 Score')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('repr_comparison_plot.png')
    
    print("\nResults saved to repr_comparison_results.csv and repr_comparison_plot.png")

if __name__ == "__main__":
    main()
