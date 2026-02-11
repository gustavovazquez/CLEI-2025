from .hdc_utils import bind, bundle, generate_library, generate_level_library, permute
from .centrality import calculate_centrality, rank_nodes
import numpy as np

class BaseEncoder:
    def __init__(self, dim=10000, repr_type='binary'):
        self.dim = dim
        self.repr_type = repr_type
        self.library = None
        self.max_nodes = 0

    def prepare_library(self, graphs):
        """Finds the maximum number of nodes and unique labels, then generates libraries."""
        self.max_nodes = max(len(g.nodes) for g in graphs) if graphs else 0
        self.library = generate_library(self.max_nodes, self.dim, repr_type=self.repr_type)
        
        # Collect unique labels across all graphs
        unique_labels = set()
        for g in graphs:
            for _, data in g.nodes(data=True):
                if 'label' in data:
                    unique_labels.add(data['label'])
        
        if unique_labels:
            sorted_labels = sorted(list(unique_labels))
            label_hvs = generate_library(len(sorted_labels), self.dim, repr_type=self.repr_type)
            self.label_library = {label: hv for label, hv in zip(sorted_labels, label_hvs)}
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
            
        # 4. Graph Bundling (Nodes + Edges)
        node_vectors = list(node_to_hv.values())
        all_vectors = node_vectors + edge_vectors
        
        if not all_vectors:
            from .hdc_utils import get_hv_class
            hv_class = get_hv_class(self.repr_type)
            return hv_class(np.zeros(self.dim))
            
        return bundle(all_vectors)

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
            from .hdc_utils import get_hv_class
            hv_class = get_hv_class(self.repr_type)
            return hv_class(np.zeros(self.dim))
            
        # 3. Direct Vertex Sum (Bundling)
        return bundle(node_vectors)

class GraphHDLevelEncoder(BaseEncoder):
    """
    GraphHD Level: Centality Ranking -> Level Mapping -> Edge Binding -> Bundling
    Uses Level Hypervectors for centrality encoding.
    """
    def __init__(self, dim=10000, num_levels=1000, repr_type='binary'):
        super().__init__(dim, repr_type=repr_type)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
        # Overwrite random library with level library
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)

    def encode(self, G, centrality_metric='pagerank'):
        # 1. Rank nodes by centrality
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)
        
        # 2. Map nodes to level hypervectors
        node_to_hv = {}
        num_nodes = len(sorted_nodes)
        
        for i, node in enumerate(sorted_nodes):
            # Map rank to level index
            if num_nodes > 1:
                level_idx = i * (self.num_levels - 1) // (num_nodes - 1)
            else:
                level_idx = 0
                
            hv = self.library[level_idx]
            
            # Incorporate label if available
            node_label = G.nodes[node].get('label')
            if node_label is not None and node_label in self.label_library:
                hv = bind(hv, self.label_library[node_label])
            
            node_to_hv[node] = hv
            
        # 3. Edge Encoding (Binding)
        edge_vectors = []
        for u, v in G.edges():
            edge_hv = bind(node_to_hv[u], node_to_hv[v])
            edge_vectors.append(edge_hv)
            
        # 4. Graph Bundling
        all_vectors = list(node_to_hv.values()) + edge_vectors
        if not all_vectors:
            from .hdc_utils import get_hv_class
            hv_class = get_hv_class(self.repr_type)
            return hv_class(np.zeros(self.dim))
            
        return bundle(all_vectors)

class GraphOrderLevelEncoder(BaseEncoder):
    """
    GraphOrder Level: Centrality Ranking -> Level Mapping -> Direct Vertex Sum
    Uses Level Hypervectors for centrality encoding.
    """
    def __init__(self, dim=10000, num_levels=1000, repr_type='binary'):
        super().__init__(dim, repr_type=repr_type)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
        # Overwrite random library with level library
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)

    def encode(self, G, centrality_metric='pagerank'):
        # 1. Rank nodes
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)
        
        # 2. Map to level HVs
        node_vectors = []
        num_nodes = len(sorted_nodes)
        
        for i, node in enumerate(sorted_nodes):
            if num_nodes > 1:
                level_idx = i * (self.num_levels - 1) // (num_nodes - 1)
            else:
                level_idx = 0
                
            hv = self.library[level_idx]
            
            node_label = G.nodes[node].get('label')
            if node_label is not None and node_label in self.label_library:
                hv = bind(hv, self.label_library[node_label])
                
            node_vectors.append(hv)
            
        if not node_vectors:
            from .hdc_utils import get_hv_class
            hv_class = get_hv_class(self.repr_type)
            return hv_class(np.zeros(self.dim))
            
        # 3. Bundling
        return bundle(node_vectors)

class GraphHDLevelPermEncoder(BaseEncoder):
    """
    GraphHD Level with Permutation: Uses permute in edge binding to avoid identity collapse.
    """
    def __init__(self, dim=10000, num_levels=1000, repr_type='binary'):
        super().__init__(dim, repr_type=repr_type)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
        # Overwrite random library with level library
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)

    def encode(self, G, centrality_metric='pagerank'):
        # 1. Rank nodes by centrality
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)
        
        # 2. Map nodes to level hypervectors
        node_to_hv = {}
        num_nodes = len(sorted_nodes)
        
        for i, node in enumerate(sorted_nodes):
            # Map rank to level index
            if num_nodes > 1:
                level_idx = i * (self.num_levels - 1) // (num_nodes - 1)
            else:
                level_idx = 0
                
            hv = self.library[level_idx]
            
            # Incorporate label if available
            node_label = G.nodes[node].get('label')
            if node_label is not None and node_label in self.label_library:
                hv = bind(hv, self.label_library[node_label])
            
            node_to_hv[node] = hv
            
        # 3. Edge Encoding with Permutation
        edge_vectors = []
        for u, v in G.edges():
            edge_hv = bind(permute(node_to_hv[u]), node_to_hv[v])
            edge_vectors.append(edge_hv)
            
        # 4. Graph Bundling
        node_vectors = list(node_to_hv.values())
        all_vectors = node_vectors + edge_vectors
        if not all_vectors:
            from .hdc_utils import get_hv_class
            hv_class = get_hv_class(self.repr_type)
            return hv_class(np.zeros(self.dim))
            
        return bundle(all_vectors)
