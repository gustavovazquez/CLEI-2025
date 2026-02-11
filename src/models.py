import numpy as np
from abc import ABC, abstractmethod

class Hypervector(ABC):
    """Abstract base class for all Hypervector representations."""
    
    def __init__(self, data):
        self.data = data
        self.dim = len(data)

    @abstractmethod
    def bind(self, other):
        """Binding operation (XOR for binary, Multiplication for float)."""
        pass

    @abstractmethod
    def bundle(self, others):
        """Bundling operation (Majority for binary, Sum/Normalization for float)."""
        pass

    @abstractmethod
    def permute(self, shift=1):
        """Cyclic shift permutation."""
        pass

    @abstractmethod
    def similarity(self, other):
        """Similarity metric (Hamming for binary, Cosine for float)."""
        pass

    @classmethod
    @abstractmethod
    def random(cls, dim):
        """Generates a random hypervector of the given dimension."""
        pass

class BinaryHypervector(Hypervector):
    """
    Binary Hypervector Representation {0, 1}^D.
    Based on HSC (Hyperdimensional Semantic Computing) / BSC (Binary Spatter Codes).
    """
    
    def bind(self, other):
        # XOR binding
        return BinaryHypervector(np.bitwise_xor(self.data, other.data))

    def bundle(self, others):
        # Majority rule
        all_data = [self.data] + [o.data for o in others]
        stacked = np.stack(all_data)
        summed = np.sum(stacked, axis=0)
        threshold = len(all_data) / 2
        # Majority: if sum > threshold -> 1, if sum < threshold -> 0. 
        # For ties, we can pick random or 0.
        result = (summed > threshold).astype(np.int8)
        # Handle ties randomly to maintain properties
        ties = summed == threshold
        if np.any(ties):
            result[ties] = np.random.choice([0, 1], size=np.sum(ties))
        return BinaryHypervector(result)

    def permute(self, shift=1):
        return BinaryHypervector(np.roll(self.data, shift))

    def similarity(self, other):
        # Hamming Similarity: 1 - HammingDistance/dim
        # Which is equivalent to (dim - XOR_sum) / dim
        xor_res = np.bitwise_xor(self.data, other.data)
        hamming_dist = np.sum(xor_res)
        return 1.0 - (hamming_dist / self.dim)

    @classmethod
    def random(cls, dim):
        return cls(np.random.randint(0, 2, size=dim, dtype=np.int8))

class FloatHypervector(Hypervector):
    """
    Real-valued Hypervector Representation R^D.
    Based on FHRR (Fractional Binding, Hardcoded Representations) or similar VSAs.
    Elements are typically drawn from a normal distribution and normalized.
    """
    
    def bind(self, other):
        # Element-wise multiplication
        return FloatHypervector(self.data * other.data)

    def bundle(self, others):
        # Normalized sum
        all_data = [self.data] + [o.data for o in others]
        summed = np.sum(all_data, axis=0)
        norm = np.linalg.norm(summed)
        if norm > 0:
            summed /= norm
        return FloatHypervector(summed.astype(np.float32))

    def permute(self, shift=1):
        return FloatHypervector(np.roll(self.data, shift))

    def similarity(self, other):
        # Cosine Similarity
        dot = np.dot(self.data, other.data)
        norm_a = np.linalg.norm(self.data)
        norm_b = np.linalg.norm(other.data)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @classmethod
    def random(cls, dim):
        # For float VSAs, elements are often i.i.d. Gaussian or Uniform
        # Here we use standard normal and normalize
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return cls(vec)

class BipolarHypervector(Hypervector):
    """
    Bipolar Hypervector Representation {-1, 1}^D.
    Commonly used in MAP (Multiply-Add-Permute) and SDM.
    """
    
    def bind(self, other):
        # Multiplication is the binding operator for bipolar vectors
        return BipolarHypervector(self.data * other.data)

    def bundle(self, others):
        # Sum and then sign
        all_data = [self.data] + [o.data for o in others]
        stacked = np.stack(all_data)
        summed = np.sum(stacked, axis=0)
        # Sign function: sum >= 0 -> 1, sum < 0 -> -1
        # Handle ties randomly
        result = np.where(summed > 0, 1, -1).astype(np.int8)
        ties = summed == 0
        if np.any(ties):
            result[ties] = np.random.choice([-1, 1], size=np.sum(ties))
        return BipolarHypervector(result)

    def permute(self, shift=1):
        return BipolarHypervector(np.roll(self.data, shift))

    def similarity(self, other):
        # Cosine Similarity is efficient for bipolar vectors: DotProduct / D
        dot = np.dot(self.data.astype(np.float32), other.data.astype(np.float32))
        return dot / self.dim

    @classmethod
    def random(cls, dim):
        return cls(np.random.choice([-1, 1], size=dim).astype(np.int8))
