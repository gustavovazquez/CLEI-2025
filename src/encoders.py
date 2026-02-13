from .hdc_utils import bind, bundle, sign, generate_library, generate_level_library, permute
from .centrality import calculate_centrality, rank_nodes
import numpy as np

class BaseEncoder:
    def __init__(self, dim=10000):
        self.dim = dim
        self.library = None
        self.max_nodes = 0

    def prepare_library(self, graphs):
        """Finds the maximum number of nodes in any graph and generates a library."""
        self.max_nodes = max(len(g.nodes) for g in graphs) if graphs else 0
        self.library = generate_library(self.max_nodes, self.dim)

class GraphHDEncoder(BaseEncoder):
    """Original GraphHD: Centrality Ranking -> Node Mapping -> Edge Binding -> Bundling"""
    
    def encode(self, G, centrality_metric='pagerank'):
        # 1. Rank nodes by centrality
        centrality = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality)
        
        # 2. Map nodes to hypervectors
        node_to_hv = {}
        for i, node in enumerate(sorted_nodes):
            node_to_hv[node] = self.library[i]
            
        # 3. Edge Encoding (Binding)
        edge_vectors = []
        for u, v in G.edges():
            edge_hv = bind(node_to_hv[u], node_to_hv[v])
            edge_vectors.append(edge_hv)
            
        if not edge_vectors and not node_to_hv:
            return np.zeros(self.dim, dtype=np.int8)
            
        # 4. Graph Bundling
        all_vectors = list(node_to_hv.values()) + edge_vectors
        graph_vector = bundle(all_vectors)
        return sign(graph_vector)

class GraphHDOrderEncoder(BaseEncoder):
    """GraphOrder: Centrality Ranking -> Node Mapping -> Direct Vertex Sum"""
    
    def encode(self, G, centrality_metric='pagerank'):
        # 1. Rank nodes by centrality
        centrality = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality)
        
        # 2. Map nodes to hypervectors
        node_vectors = []
        for i, node in enumerate(sorted_nodes):
            node_vectors.append(self.library[i])
            
        if not node_vectors:
            return np.zeros(self.dim, dtype=np.int8)
            
        # 3. Direct Vertex Sum (Bundling)
        graph_vector = bundle(node_vectors)
        return sign(graph_vector)

class GraphHDLevelEncoder(BaseEncoder):
    """GraphHD Level: Uses level-based hypervectors with permutation for edge encoding"""
    def __init__(self, dim=10000, num_levels=1000):
        super().__init__(dim)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        self.library = generate_level_library(self.num_levels, self.dim)

    def encode(self, G, centrality_metric='pagerank'):
        centrality = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality)
        node_to_hv = {}
        num_nodes = len(sorted_nodes)
        
        for i, node in enumerate(sorted_nodes):
            level_idx = i * (self.num_levels - 1) // (num_nodes - 1) if num_nodes > 1 else 0
            node_to_hv[node] = self.library[level_idx]
            
        edge_vectors = [bind(permute(node_to_hv[u]), node_to_hv[v]) for u, v in G.edges()]
        all_vectors = list(node_to_hv.values()) + edge_vectors
        return sign(bundle(all_vectors)) if all_vectors else np.zeros(self.dim, dtype=np.int8)
