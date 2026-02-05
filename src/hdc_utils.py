import numpy as np

def generate_hypervector(dim):
    """Generates a random bipolar hypervector {-1, 1}^dim."""
    return np.random.choice([-1, 1], size=dim).astype(np.int8)

def generate_library(num_vectors, dim):
    """Generates a library of random hypervectors."""
    return [generate_hypervector(dim) for _ in range(num_vectors)]

def bind(hv1, hv2):
    """Binds two hypervectors using element-wise multiplication."""
    return hv1 * hv2

def bundle(hvs):
    """Bundles multiple hypervectors using element-wise sum with promotion to avoid overflow."""
    if not hvs:
        return None
    # Promote to int64 to prevent overflow during sum of many int8 vectors
    return np.sum(hvs, axis=0, dtype=np.int64)

def sign(hv):
    """Applies the sign function to a hypervector to map it back to {-1, 1}."""
    return np.where(hv >= 0, 1, -1).astype(np.int8)

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors. Casts to float to avoid overflow."""
    v1_f = v1.astype(np.float64)
    v2_f = v2.astype(np.float64)
    dot_product = np.dot(v1_f, v2_f)
    norm_v1 = np.linalg.norm(v1_f)
    norm_v2 = np.linalg.norm(v2_f)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)
