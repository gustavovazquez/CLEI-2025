import numpy as np
from src.hdc_utils import generate_level_library, cos_similarity
from src.encoders import GraphHDLevelEncoder
from src.models import BinaryHypervector
import networkx as nx

def test_level_similarity():
    print("Testing Level Hypervector Similarity...")
    dim = 10000
    num_levels = 100
    # Uses default binary representation
    library = generate_level_library(num_levels, dim, repr_type='binary')
    
    # Check similarity between Level 0 and others
    similarities = [library[0].similarity(library[i]) for i in range(num_levels)]
    
    print(f"Similarity Level 0 to 1: {similarities[1]:.4f}")
    print(f"Similarity Level 0 to 50: {similarities[50]:.4f}")
    print(f"Similarity Level 0 to 99: {similarities[99]:.4f}")
    
    # Check if monotonically decreasing (mostly)
    is_decreasing = all(similarities[i] >= similarities[i+1] for i in range(num_levels-1))
    print(f"Monotonically decreasing: {is_decreasing}")
    
    if not is_decreasing:
        # It might not be perfectly monotonic due to random sample, but should be generally decreasing
        avg_diff = np.mean([similarities[i] - similarities[i+1] for i in range(num_levels-1)])
        print(f"Average similarity decrease per level: {avg_diff:.6f}")

def test_encoder_run():
    print("\nTesting Encoder Execution...")
    G = nx.erdos_renyi_graph(10, 0.5)
    # Add dummy labels
    for n in G.nodes:
        G.nodes[n]['label'] = np.random.randint(0, 3)
        
    encoder = GraphHDLevelEncoder(dim=10000, num_levels=10, repr_type='binary')
    # Mock prepare_library
    encoder.label_library = {
        0: BinaryHypervector.random(10000), 
        1: BinaryHypervector.random(10000), 
        2: BinaryHypervector.random(10000)
    }
    encoder.library = generate_level_library(10, 10000, repr_type='binary')
    
    try:
        hv = encoder.encode(G, centrality_metric='pagerank')
        print(f"Encoded Graph HV dim: {hv.dim}")
        print("Encoder execution successful.")
    except Exception as e:
        print(f"Encoder execution failed: {e}")

if __name__ == "__main__":
    test_level_similarity()
    test_encoder_run()
