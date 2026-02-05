from .hdc_utils import bind, bundle, sign, generate_library
from .centrality import calculate_centrality, rank_nodes
import numpy as np

class BaseEncoder:
    def __init__(self, dim=10000):
        self.dim = dim
        self.library = None
        self.max_nodes = 0

    def prepare_library(self, graphs):
        """Finds the maximum number of nodes in any graph and generates a library."""
        self.max_nodes = max(len(g.nodes) for g in graphs)
        self.library = generate_library(self.max_nodes, self.dim)

class GraphHDEncoder(BaseEncoder):
    """Original GraphHD: Centrality Ranking -> Node Mapping -> Edge Binding -> Bundling"""
    
    def encode(self, G, centrality_metric='pagerank'):
        # 1. Rank nodes by centrality
        centrality = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality)
        
        # 2. Map nodes to hypervectors (mapping rank to library index)
        node_to_hv = {}
        for i, node in enumerate(sorted_nodes):
            node_to_hv[node] = self.library[i]
            
        # 3. Edge Encoding (Binding)
        edge_vectors = []
        for u, v in G.edges():
            edge_hv = bind(node_to_hv[u], node_to_hv[v])
            edge_vectors.append(edge_hv)
            
        if not edge_vectors:
            return np.zeros(self.dim, dtype=np.int8)
            
        # 4. Graph Bundling
        graph_vector = bundle(edge_vectors)
        return sign(graph_vector)

class GraphOrderEncoder(BaseEncoder):
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
