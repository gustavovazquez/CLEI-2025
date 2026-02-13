# Graph Classification with Hyperdimensional Computing

This repository contains the implementation of **GraphHD** and variants for graph classification proposed in **Sica&Vazquez (2025)** using Hyperdimensional Computing (HDC). The code supports multiple centrality-based encoding methods and provides a flexible benchmarking framework.

## Reference

This implementation is based on the following papers:

- **Original GraphHD**: Morris, C., et al. (2022). "GraphHD: Efficient graph classification using hyperdimensional computing." *arXiv preprint arXiv:2205.07826*.

- **GraphHD Variants**: Vazquez, G. & Sica, I. (October 2025). "Exploring Centrality Measures and Encoding Variants for Graph Classification in Hyperdimensional Computing." *51 CLEI*.


## Supplementary material
File final_benchmark_results.csv contains the results of the experiments of Sica&Vazquez (2025).

## Implemented Methods

### 1. **GraphHD** (Original)
The original GraphHD method that uses:
- Centrality-based node ranking
- Node-to-hypervector mapping
- Edge binding operations
- Graph-level bundling

### 2. **GraphHD-Order**
A simplified variant that:
- Ranks nodes by centrality
- Maps nodes to hypervectors
- Directly bundles node vectors (no edge binding)

### 3. **GraphHD-Level**
An enhanced variant that:
- Uses level-based hypervector libraries
- Maps centrality values to continuous levels

## Installation

### Prerequisites
- Python 3.8+
- NumPy
- NetworkX
- scikit-learn
- pandas
- matplotlib (for visualization)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/graphhd-clei2025.git
cd graphhd-clei2025

# Install dependencies
pip install numpy networkx scikit-learn pandas matplotlib
```

## Datasets

The implementation supports TUDataset benchmark datasets:
- MUTAG
- PROTEINS
- ENZYMES
- NCI1
- DD
- PTC_FM

Datasets are automatically downloaded to the `./data` directory on first use.

## Usage

### Basic Usage

Run a simple experiment with default parameters:

```bash
python main.py --dataset MUTAG --n_rep 10
```

### Parallel Benchmarking

The `benchmark_parallel.py` script allows flexible experiment configuration:

#### Run all encoders on MUTAG with Katz centrality (5 repetitions)
```bash
python benchmark_parallel.py --datasets MUTAG --metrics katz --n_rep 5 --output mutag_katz_results.csv
```

#### Run specific encoder on multiple datasets
```bash
python benchmark_parallel.py --datasets MUTAG PROTEINS --encoders GraphHD --n_rep 10
```

#### Run all combinations (default: 1 repetition)
```bash
python benchmark_parallel.py --n_rep 100 --output full_benchmark.csv
```

### Command-Line Arguments

#### `benchmark_parallel.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datasets` | str+ | All | Dataset(s) to run: MUTAG, PROTEINS, ENZYMES, NCI1, DD, PTC_FM |
| `--encoders` | str+ | All | Encoder(s): GraphHD, GraphOrder, GraphHDLevel |
| `--metrics` | str+ | All | Centrality metrics: pagerank, degree, closeness, betweenness, eigenvector, katz |
| `--n_rep` | int | 1 | Number of repetitions per experiment |
| `--dim` | int | 10000 | Hypervector dimension |
| `--workers` | int | CPU count | Number of parallel workers |
| `--output` | str | benchmark_parallel_results.csv | Output CSV file |
| `--data_dir` | str | ./data | Data directory |

#### `main.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | MUTAG | Dataset name |
| `--dim` | int | 10000 | Hypervector dimension |
| `--n_rep` | int | 10 | Number of repetitions |
| `--metrics` | str+ | pagerank, degree, closeness | Centrality metrics to test |
| `--data_dir` | str | ./data | Data directory |

## Output Format

Results are saved as CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `Dataset` | Dataset name |
| `Encoder` | Encoding method (GraphHD, GraphOrder, GraphHDLevel) |
| `Centrality` | Centrality metric used |
| `f1_mean` | Mean F1 score across repetitions |
| `f1_std` | Standard deviation of F1 score |
| `accuracy_mean` | Mean accuracy across repetitions |
| `accuracy_std` | Standard deviation of accuracy |
| `total_time_sec` | Total execution time in seconds |

### Example Output

```csv
Dataset,Encoder,Centrality,f1_mean,f1_std,accuracy_mean,accuracy_std,total_time_sec
MUTAG,GraphHD,katz,0.8755,0.0498,0.8737,0.0510,0.62
MUTAG,GraphOrder,katz,0.8381,0.0605,0.8368,0.0586,0.30
MUTAG,GraphHDLevel,katz,0.7713,0.0781,0.7842,0.0788,1.67
```

## Project Structure

```
graphhd-clei2025/
├── src/
│   ├── encoders.py          # GraphHD encoder implementations
│   ├── hdc_utils.py          # HDC operations (bind, bundle, permute)
│   ├── centrality.py         # Centrality measure calculations
│   ├── experiments.py        # Experiment runner with cross-validation
│   ├── dataset_loader.py     # TUDataset loader
│   └── visualization.py      # Result visualization utilities
├── benchmark_parallel.py     # Parallel benchmark script
├── benchmark_comprehensive.py # Comprehensive benchmark
├── main.py                   # Simple experiment runner
├── data/                     # Dataset directory (auto-created)
└── README.md
```

## Centrality Measures

The implementation supports six centrality measures:

1. **PageRank**: Importance based on graph structure
2. **Degree**: Number of connections per node
3. **Closeness**: Average distance to all other nodes
4. **Betweenness**: Number of shortest paths through a node
5. **Eigenvector**: Influence based on connections to important nodes
6. **Katz**: Weighted sum of all path lengths

## Example Experiments

### Experiment 1: Compare all encoders on MUTAG
```bash
python benchmark_parallel.py --datasets MUTAG --n_rep 100 --output mutag_comparison.csv
```

### Experiment 2: Test Katz centrality across all datasets
```bash
python benchmark_parallel.py --metrics katz --n_rep 50 --output katz_benchmark.csv
```

### Experiment 3: Single encoder, all metrics
```bash
python benchmark_parallel.py --datasets PROTEINS --encoders GraphHD --n_rep 20
```

## Performance Notes

- **GraphHD**: Best accuracy, moderate speed
- **GraphHD-Order**: Fastest execution, good accuracy
- **GraphHD-Level**: Slower but uses continuous centrality values

Parallel execution uses all available CPU cores by default. Adjust with `--workers` for resource management.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{vazquez2025graphhd,
  title={Exploring Centrality Measures and Encoding Variants for Graph Classification in Hyperdimensional Computing},
  author={Vazquez, Gustavo and Sica, Ignacio},
  booktitle={51 CLEI},
  year={2025},
  month={October}
}
```

## License

This project is available for academic and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgement

We gratefully thank to Agencia Nacional de Investigación e Innovación (ANII-Uruguay) for funding this work through grant number FCE-1-2023-1-176242.

## Contact

For questions or collaborations, please contact the authors through the CLEI 2025 conference proceedings.

