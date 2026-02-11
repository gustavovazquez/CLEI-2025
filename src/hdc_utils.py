import numpy as np
from .models import BinaryHypervector, FloatHypervector, BipolarHypervector

def get_hv_class(repr_type='binary'):
    """Returns the hypervector class based on the representation type."""
    if repr_type == 'binary':
        return BinaryHypervector
    elif repr_type == 'float':
        return FloatHypervector
    elif repr_type == 'bipolar':
        return BipolarHypervector
    else:
        raise ValueError(f"Unknown representation type: {repr_type}")

def generate_hypervector(dim, repr_type='binary'):
    """Generates a random hypervector of the given dimension and type."""
    hv_class = get_hv_class(repr_type)
    return hv_class.random(dim)

def generate_library(num_vectors, dim, repr_type='binary'):
    """Generates a library of random hypervectors."""
    return [generate_hypervector(dim, repr_type) for _ in range(num_vectors)]

def generate_level_library(num_levels, dim, repr_type='binary'):
    """
    Generates a library of level hypervectors.
    Adjacent levels share high similarity.
    """
    hv_class = get_hv_class(repr_type)
    library = [hv_class.random(dim)]
    
    # Define how to "step" to the next level based on representation
    if repr_type == 'binary':
        # Flip bits to go from random to orthogonal (dim/2 bits in total)
        bits_to_flip = dim // (2 * (num_levels - 1))
        if bits_to_flip == 0: bits_to_flip = 1
        
        for i in range(1, num_levels):
            prev_data = library[-1].data.copy()
            flip_indices = np.random.choice(dim, size=bits_to_flip, replace=False)
            prev_data[flip_indices] = 1 - prev_data[flip_indices]
            library.append(BinaryHypervector(prev_data))
            
    elif repr_type == 'float':
        # For float, we can blend with a new random vector to slowly diverge
        # Correlation between levels: 1 -> 0
        for i in range(1, num_levels):
            prev_data = library[-1].data
            new_random = np.random.randn(dim).astype(np.float32)
            new_random /= np.linalg.norm(new_random)
            # Interpolation factor: we want to reach orthogonality (0 similarity) in num_levels
            # A simple way: add a bit of noise and re-normalize
            # alpha = step_size
            alpha = 1.0 / (num_levels - 1)
            next_data = (1 - alpha) * prev_data + alpha * new_random
            next_data /= np.linalg.norm(next_data)
            library.append(FloatHypervector(next_data))
            
    elif repr_type == 'bipolar':
        # Similar to binary, flip -1 to 1 to go from random to orthogonal
        bits_to_flip = dim // (2 * (num_levels - 1))
        if bits_to_flip == 0: bits_to_flip = 1
        
        for i in range(1, num_levels):
            prev_data = library[-1].data.copy()
            flip_indices = np.random.choice(dim, size=bits_to_flip, replace=False)
            prev_data[flip_indices] *= -1
            library.append(BipolarHypervector(prev_data))
            
    return library

# Wrapper functions for backward compatibility if needed, 
# but recommendation is to use the methods in Hypervector objects.

def bind(hv1, hv2):
    return hv1.bind(hv2)

def bundle(hvs):
    if not hvs: return None
    if len(hvs) == 1: return hvs[0]
    return hvs[0].bundle(hvs[1:])

def permute(hv, shift=1):
    return hv.permute(shift)

def cos_similarity(hv1, hv2):
    return hv1.similarity(hv2)
