<<<<<<< HEAD
import torch
from .hdc_utils import bind, bundle, generate_library, generate_level_library, permute, get_hv_class
=======
from .hdc_utils import bind, bundle, sign, generate_library, generate_level_library, permute
>>>>>>> parent of 344d83b (scripts finales)
from .centrality import calculate_centrality, rank_nodes
from .models import BinaryHypervector, DEVICE


class BaseEncoder:
    def __init__(self, dim=10000):
        self.dim = dim
        self.library = None
        self.max_nodes = 0

    def prepare_library(self, graphs):
        """Finds the maximum number of nodes and unique labels, then generates libraries."""
<<<<<<< HEAD
        self.max_nodes = max(len(g.nodes) for g in graphs) if graphs else 0
        self.library = generate_library(self.max_nodes, self.dim, repr_type=self.repr_type)

=======
        self.max_nodes = max(len(g.nodes) for g in graphs)
        self.library = generate_library(self.max_nodes, self.dim)
        
        # Collect unique labels across all graphs
>>>>>>> parent of 344d83b (scripts finales)
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

    def _map_nodes(self, G, centrality_metric):
        """Shared: rank nodes and map to HVs (with label binding)."""
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)

        node_to_hv = {}
        for i, node in enumerate(sorted_nodes):
            hv = self.library[i]
            node_label = G.nodes[node].get('label')
            if node_label is not None and node_label in self.label_library:
                hv = bind(hv, self.label_library[node_label])
            node_to_hv[node] = hv
<<<<<<< HEAD

        return node_to_hv

    def _batch_edge_bind(self, G, node_to_hv, use_permute=False):
        """Batch edge binding on GPU: bind all edges in one vectorized op."""
        if G.number_of_edges() == 0:
            return None

        edges = list(G.edges())
        src_data = torch.stack([node_to_hv[u].data for u, _ in edges])
        dst_data = torch.stack([node_to_hv[v].data for _, v in edges])

        if use_permute:
            src_data = torch.roll(src_data, shifts=1, dims=1)

        if isinstance(self.library[0], BinaryHypervector):
            return torch.bitwise_xor(src_data, dst_data)
        else:
            return src_data * dst_data

    def _bundle_all(self, node_to_hv, edge_data=None):
        """Bundle node + edge tensors using batch_bundle on GPU."""
        hv_class = type(self.library[0])
        node_data = torch.stack([hv.data for hv in node_to_hv.values()])

        if edge_data is not None:
            all_data = torch.cat([node_data, edge_data], dim=0)
        else:
            all_data = node_data

        return hv_class.batch_bundle(all_data)


class GraphHDEncoder(BaseEncoder):
    """Original GraphHD: Centrality Ranking -> Node Mapping -> Edge Binding -> Bundling"""

    def encode(self, G, centrality_metric='pagerank'):
        node_to_hv = self._map_nodes(G, centrality_metric)

        if not node_to_hv:
            hv_class = get_hv_class(self.repr_type)
            return hv_class(torch.zeros(self.dim, device=DEVICE))

        edge_data = self._batch_edge_bind(G, node_to_hv)
        return self._bundle_all(node_to_hv, edge_data)

=======
            
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
>>>>>>> parent of 344d83b (scripts finales)

class GraphOrderEncoder(BaseEncoder):
    """GraphOrder: Centrality Ranking -> Node Mapping -> Direct Vertex Sum"""

    def encode(self, G, centrality_metric='pagerank'):
<<<<<<< HEAD
        node_to_hv = self._map_nodes(G, centrality_metric)

        if not node_to_hv:
            hv_class = get_hv_class(self.repr_type)
            return hv_class(torch.zeros(self.dim, device=DEVICE))

        hv_class = type(self.library[0])
        node_data = torch.stack([hv.data for hv in node_to_hv.values()])
        return hv_class.batch_bundle(node_data)


class GraphHDLevelEncoder(BaseEncoder):
    """GraphHD Level: Level HVs for centrality encoding + Edge Binding."""

    def __init__(self, dim=10000, num_levels=1000, repr_type='binary'):
        super().__init__(dim, repr_type=repr_type)
=======
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

class GraphHDLevelEncoder(BaseEncoder):
    """
    GraphHD Level: Centality Ranking -> Level Mapping -> Edge Binding -> Bundling
    Uses Level Hypervectors for centrality encoding.
    """
    def __init__(self, dim=10000, num_levels=100):
        super().__init__(dim)
>>>>>>> parent of 344d83b (scripts finales)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
<<<<<<< HEAD
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)
=======
        # Overwrite random library with level library
        self.library = generate_level_library(self.num_levels, self.dim)
>>>>>>> parent of 344d83b (scripts finales)

    def _map_nodes_level(self, G, centrality_metric):
        """Map nodes using level HVs instead of random HVs."""
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)

        node_to_hv = {}
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
            node_to_hv[node] = hv
<<<<<<< HEAD

        return node_to_hv

    def encode(self, G, centrality_metric='pagerank'):
        node_to_hv = self._map_nodes_level(G, centrality_metric)

        if not node_to_hv:
            hv_class = get_hv_class(self.repr_type)
            return hv_class(torch.zeros(self.dim, device=DEVICE))

        edge_data = self._batch_edge_bind(G, node_to_hv)
        return self._bundle_all(node_to_hv, edge_data)


class GraphOrderLevelEncoder(BaseEncoder):
    """GraphOrder Level: Level HVs + Direct Vertex Sum."""

    def __init__(self, dim=10000, num_levels=1000, repr_type='binary'):
        super().__init__(dim, repr_type=repr_type)
=======
            
        # 3. Edge Encoding (Binding)
        edge_vectors = []
        for u, v in G.edges():
            edge_hv = bind(node_to_hv[u], node_to_hv[v])
            edge_vectors.append(edge_hv)
            
        if not edge_vectors:
            node_vectors = list(node_to_hv.values())
            if not node_vectors:
                return np.zeros(self.dim, dtype=np.int8)
            return sign(bundle(node_vectors))
            
        # 4. Graph Bundling
        all_vectors = list(node_to_hv.values()) + edge_vectors
        return sign(bundle(all_vectors))

class GraphOrderLevelEncoder(BaseEncoder):
    """
    GraphOrder Level: Centrality Ranking -> Level Mapping -> Direct Vertex Sum
    Uses Level Hypervectors for centrality encoding.
    """
    def __init__(self, dim=10000, num_levels=100):
        super().__init__(dim)
>>>>>>> parent of 344d83b (scripts finales)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
<<<<<<< HEAD
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)
=======
        # Overwrite random library with level library
        self.library = generate_level_library(self.num_levels, self.dim)
>>>>>>> parent of 344d83b (scripts finales)

    def encode(self, G, centrality_metric='pagerank'):
        # Reuse level mapping from GraphHDLevelEncoder
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)

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
<<<<<<< HEAD
            hv_class = get_hv_class(self.repr_type)
            return hv_class(torch.zeros(self.dim, device=DEVICE))

        hv_class = type(self.library[0])
        node_data = torch.stack([hv.data for hv in node_vectors])
        return hv_class.batch_bundle(node_data)


class GraphHDLevelPermEncoder(BaseEncoder):
    """GraphHD Level with Permutation: permute in edge binding to avoid identity collapse."""

    def __init__(self, dim=10000, num_levels=1000, repr_type='binary'):
        super().__init__(dim, repr_type=repr_type)
=======
            return np.zeros(self.dim, dtype=np.int8)
            
        # 3. Bundling
        return sign(bundle(node_vectors))

class GraphOrderLevelEncoder(BaseEncoder):
    """
    GraphOrder Level: Centrality Ranking -> Level Mapping -> Direct Vertex Sum
    Uses Level Hypervectors for centrality encoding.
    """
    def __init__(self, dim=10000, num_levels=100):
        super().__init__(dim)
>>>>>>> parent of 344d83b (scripts finales)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
<<<<<<< HEAD
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)
=======
        # Overwrite random library with level library
        self.library = generate_level_library(self.num_levels, self.dim)

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
            return np.zeros(self.dim, dtype=np.int8)
            
        # 3. Bundling
        return sign(bundle(node_vectors))

class GraphHDLevelPermEncoder(BaseEncoder):
    """
    GraphHD Level with Permutation: Uses permute in edge binding to avoid identity collapse.
    This encoder fixes the issue where similar Level HVs produce near-identity edge vectors.
    """
    def __init__(self, dim=10000, num_levels=100):
        super().__init__(dim)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
        # Overwrite random library with level library
        self.library = generate_level_library(self.num_levels, self.dim)
>>>>>>> parent of 344d83b (scripts finales)

    def encode(self, G, centrality_metric='pagerank'):
        centrality_data = calculate_centrality(G, metric=centrality_metric)
        sorted_nodes = rank_nodes(centrality_data)

        node_to_hv = {}
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
            node_to_hv[node] = hv
<<<<<<< HEAD

        if not node_to_hv:
            hv_class = get_hv_class(self.repr_type)
            return hv_class(torch.zeros(self.dim, device=DEVICE))

        # Batch edge binding WITH permutation
        edge_data = self._batch_edge_bind(G, node_to_hv, use_permute=True)
        return self._bundle_all(node_to_hv, edge_data)
=======
            
        # 3. Edge Encoding with Permutation (fixes identity collapse)
        edge_vectors = []
        for u, v in G.edges():
            # Use permutation to break symmetry: Edge = permute(HV_u) * HV_v
            edge_hv = bind(permute(node_to_hv[u]), node_to_hv[v])
            edge_vectors.append(edge_hv)
            
        if not edge_vectors:
            node_vectors = list(node_to_hv.values())
            if not node_vectors:
                return np.zeros(self.dim, dtype=np.int8)
            return sign(bundle(node_vectors))
            
        # 4. Graph Bundling
        all_vectors = list(node_to_hv.values()) + edge_vectors
        return sign(bundle(all_vectors))
>>>>>>> parent of 344d83b (scripts finales)
