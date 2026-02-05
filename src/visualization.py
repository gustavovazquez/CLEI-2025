import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_results_table(results_list):
    """
    Creates a Pandas DataFrame from a list of result dictionaries.
    Columns: Dataset, Encoder, Centrality, F1-Mean, F1-Std, Acc-Mean, Acc-Std
    """
    df = pd.DataFrame(results_list)
    return df

def plot_comparison(df, metric='f1_mean', output_path='results_plot.png'):
    """Plots a comparison of centralities across different methods."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Centrality', y=metric, hue='Encoder')
    plt.title(f'Comparison of {metric} across Centralities and Encoders')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()
    return os.path.abspath(output_path)
