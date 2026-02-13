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
    """Bundles multiple hypervectors using element-wise sum."""
    return np.sum(hvs, axis=0)

def sign(hv):
    """Applies the sign function to a hypervector to map it back to {-1, 1}."""
    return np.where(hv >= 0, 1, -1).astype(np.int8)

def generate_level_library(num_levels, dim):
    """
    Generates a library of level hypervectors.
    Adjacent levels share a high number of bits.
    """
    library = [generate_hypervector(dim)]
    # Number of bits to flip between levels to go from random to orthogonal
    bits_to_flip = dim // (2 * (num_levels - 1))
    
    if bits_to_flip == 0:
        bits_to_flip = 1
        
    for i in range(1, num_levels):
        prev_hv = library[-1].copy()
        flip_indices = np.random.choice(dim, size=bits_to_flip, replace=False)
        prev_hv[flip_indices] *= -1
        library.append(prev_hv)
        
    return library

def permute(hv, shift=1):
    """Permutes a hypervector by cyclic shift."""
    return np.roll(hv, shift)

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(v1.astype(float), v2.astype(float))
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)
