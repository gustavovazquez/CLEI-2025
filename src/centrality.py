import networkx as nx

def calculate_centrality(G, metric='pagerank'):
    """
    Calculates centrality for all nodes in the graph G.
    Supported metrics: pagerank, degree, closeness, betweenness, katz, eigenvector.
    """
    if metric == 'pagerank':
        return nx.pagerank(G, weight=None)
    elif metric == 'degree':
        return dict(nx.degree(G))
    elif metric == 'closeness':
        return nx.closeness_centrality(G)
    elif metric == 'betweenness':
        return nx.betweenness_centrality(G)
    elif metric == 'katz':
        # Use a small alpha for Katz to ensure convergence
        try:
            return nx.katz_centrality(G, alpha=0.1, beta=1.0)
        except:
            # Fallback for small graphs or non-convergence
            return nx.degree_centrality(G)
    elif metric == 'eigenvector':
        try:
            return nx.eigenvector_centrality(G, max_iter=1000)
        except:
            # Fallback to degree for non-convergent cases
            return nx.degree_centrality(G)
    else:
        raise ValueError(f"Unsupported centrality metric: {metric}")

def rank_nodes(centrality_dict):
    """Returns nodes sorted by centrality score in descending order."""
    return sorted(centrality_dict.keys(), key=lambda x: centrality_dict[x], reverse=True)
