from .hdc_utils import bind, bundle, sign, generate_library
from .centrality import calculate_centrality, rank_nodes
import numpy as np

class BaseEncoder:
    def __init__(self, dim=10000):
        self.dim = dim
        self.library = None
        self.max_nodes = 0

    def prepare_library(self, graphs):
        """Finds the maximum number of nodes and unique labels, then generates libraries."""
        self.max_nodes = max(len(g.nodes) for g in graphs)
        self.library = generate_library(self.max_nodes, self.dim)
        
        # Collect unique labels across all graphs
        unique_labels = set()
        for g in graphs:
            for _, data in g.nodes(data=True):
                if 'label' in data:
                    unique_labels.add(data['label'])
        
        if unique_labels:
            self.label_library = {label: hv for label, hv in zip(sorted(list(unique_labels)), 
                                                                generate_library(len(unique_labels), self.dim))}
        else:
            self.label_library = {}

class GraphHDEncoder(BaseEncoder):
    """Original GraphHD: Centrality Ranking -> Node Mapping -> Edge Binding -> Bundling"""
    
    def encode(self, G, centrality_metric='pagerank'):
        # 1. Rank nodes by centrality
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)
        
        # 2. Map nodes to hypervectors
        node_to_hv = {}
        for i, node in enumerate(sorted_nodes):
            # Base HV is the rank HV
            hv = self.library[i]
            
            # If node has a label, bind it with the rank HV
            node_label = G.nodes[node].get('label')
            if node_label is not None and node_label in self.label_library:
                hv = bind(hv, self.label_library[node_label])
            
            node_to_hv[node] = hv
            
        # 3. Edge Encoding (Binding)
        edge_vectors = []
        for u, v in G.edges():
            edge_hv = bind(node_to_hv[u], node_to_hv[v])
            edge_vectors.append(edge_hv)
            
        if not edge_vectors:
            # If no edges, fallback to bundling node vectors
            node_vectors = list(node_to_hv.values())
            if not node_vectors:
                return np.zeros(self.dim, dtype=np.int8)
            return sign(bundle(node_vectors))
            
        # 4. Graph Bundling (Nodes + Edges)
        node_vectors = list(node_to_hv.values())
        all_vectors = node_vectors + edge_vectors
        graph_vector = bundle(all_vectors)
        return sign(graph_vector)

class GraphOrderEncoder(BaseEncoder):
    """GraphOrder: Centrality Ranking -> Node Mapping -> Direct Vertex Sum"""
    
    def encode(self, G, centrality_metric='pagerank'):
        # 1. Rank nodes by centrality
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)
        
        # 2. Map nodes to hypervectors
        node_vectors = []
        for i, node in enumerate(sorted_nodes):
            hv = self.library[i]
            
            # Incorporate label if available
            node_label = G.nodes[node].get('label')
            if node_label is not None and node_label in self.label_library:
                hv = bind(hv, self.label_library[node_label])
                
            node_vectors.append(hv)
            
        if not node_vectors:
            return np.zeros(self.dim, dtype=np.int8)
            
        # 3. Direct Vertex Sum (Bundling)
        graph_vector = bundle(node_vectors)
        return sign(graph_vector)
