import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_results_table(results_list):
    """
    Creates a Pandas DataFrame from a list of result dictionaries.
    Columns: Dataset, Encoder, Centrality, F1-Mean, F1-Std, Acc-Mean, Acc-Std, Time-Total
    """
    df = pd.DataFrame(results_list)
    return df

def plot_comparison(df, metric='f1_mean', output_path='results_plot.png'):
    """Plots a comparison of centralities across different methods."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Centrality', y=metric, hue='Encoder')
    plt.title(f'Comparison of {metric} across Centralities and Encoders')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()
    return os.path.abspath(output_path)

def plot_times(df, output_path='times_plot.png'):
    """Plots a comparison of execution times."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Centrality', y='total_time_sec', hue='Encoder')
    plt.title('Execution Time Comparison (Total for N Repetitions)')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()
    return os.path.abspath(output_path)
