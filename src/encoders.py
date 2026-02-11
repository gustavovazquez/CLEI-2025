import torch
from .hdc_utils import bind, bundle, generate_library, generate_level_library, permute, get_hv_class
from .centrality import calculate_centrality, rank_nodes
from .models import BinaryHypervector, DEVICE


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


class GraphOrderEncoder(BaseEncoder):
    """GraphOrder: Centrality Ranking -> Node Mapping -> Direct Vertex Sum"""

    def encode(self, G, centrality_metric='pagerank'):
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
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)

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
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)

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
            hv_class = get_hv_class(self.repr_type)
            return hv_class(torch.zeros(self.dim, device=DEVICE))

        hv_class = type(self.library[0])
        node_data = torch.stack([hv.data for hv in node_vectors])
        return hv_class.batch_bundle(node_data)


class GraphHDLevelPermEncoder(BaseEncoder):
    """GraphHD Level with Permutation: permute in edge binding to avoid identity collapse."""

    def __init__(self, dim=10000, num_levels=1000, repr_type='binary'):
        super().__init__(dim, repr_type=repr_type)
        self.num_levels = num_levels

    def prepare_library(self, graphs):
        super().prepare_library(graphs)
        self.library = generate_level_library(self.num_levels, self.dim, repr_type=self.repr_type)

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

        if not node_to_hv:
            hv_class = get_hv_class(self.repr_type)
            return hv_class(torch.zeros(self.dim, device=DEVICE))

        # Batch edge binding WITH permutation
        edge_data = self._batch_edge_bind(G, node_to_hv, use_permute=True)
        return self._bundle_all(node_to_hv, edge_data)
