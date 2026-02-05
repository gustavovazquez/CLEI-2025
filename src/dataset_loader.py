import os
import networkx as nx

def load_tudataset(data_dir, dataset_name):
    """
    Loads a TUDataset from the specified directory.
    Returns: list of NetworkX graphs, list of labels.
    """
    base_path = os.path.join(data_dir, dataset_name, dataset_name)
    
    # 1. Load graph indicator (node id to graph id)
    # TUDataset graph ids are typically 1-indexed
    with open(f"{base_path}_graph_indicator.txt", 'r') as f:
        node_to_graph = [int(line.strip()) for line in f]
        
    num_nodes = len(node_to_graph)
    num_graphs = max(node_to_graph)
    
    graphs = [nx.Graph() for _ in range(num_graphs)]
    
    # Add nodes to graphs
    for i, graph_id in enumerate(node_to_graph):
        # node index is i+1 in A.txt usually, but we can just use i as identifier within graph
        # mapping node index i to a local index within its graph
        graphs[graph_id-1].add_node(i+1)
        
    # 2. Load adjacency matrix (edges)
    with open(f"{base_path}_A.txt", 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split(','))
            # u and v are global node ids
            g_u = node_to_graph[u-1]
            g_v = node_to_graph[v-1]
            if g_u == g_v:
                graphs[g_u-1].add_edge(u, v)
    
    # 3. Load graph labels
    with open(f"{base_path}_graph_labels.txt", 'r') as f:
        labels = [int(line.strip()) for line in f]
        
    # 4. Load node labels if they exist
    node_labels = None
    node_labels_path = f"{base_path}_node_labels.txt"
    if os.path.exists(node_labels_path):
        with open(node_labels_path, 'r') as f:
            node_labels = [int(line.strip()) for line in f]
        
        # Add labels as node attributes
        for i, label in enumerate(node_labels):
            g_id = node_to_graph[i]
            # Nodes are i+1
            graphs[g_id-1].nodes[i+1]['label'] = label
            
    return graphs, labels
