import networkx as nx

def calculate_centrality(G, metric='pagerank'):
    """
    Calculates primary centrality and returns stable signatures for nodes.
    """
    if metric == 'pagerank':
        primary = nx.pagerank(G, weight=None, max_iter=200)
    elif metric == 'degree':
        primary = dict(nx.degree(G))
    elif metric == 'closeness':
        primary = nx.closeness_centrality(G)
    elif metric == 'betweenness':
        primary = nx.betweenness_centrality(G)
    else:
        primary = dict(nx.degree(G))
        
    # Always include degree and labels for tie-breaking
    degree = dict(nx.degree(G))
    labels = {n: G.nodes[n].get('label', 0) for n in G.nodes}
    
    # WL-inspired signature for stable tie-breaking:
    # Hash of (label, degree, sorted-neighbor-labels)
    signatures = {}
    for n in G.nodes():
        neighbor_labels = sorted([G.nodes[m].get('label', 0) for m in G.neighbors(n)])
        signatures[n] = (labels[n], degree[n], tuple(neighbor_labels))
    
    return primary, signatures

def rank_nodes(centrality_data):
    """Returns nodes sorted by (primary_cent, signature) to maximize stability."""
    primary, signatures = centrality_data
    nodes = list(primary.keys())
    # Sort by descending primary, then by the signature tuple
    return sorted(nodes, key=lambda x: (primary[x], signatures[x]), reverse=True)
