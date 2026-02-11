import os
import torch
from abc import ABC, abstractmethod

# Auto-detect device (set HDC_FORCE_CPU=1 to force CPU)
_force_cpu = os.environ.get('HDC_FORCE_CPU', '0') == '1'
DEVICE = torch.device('cpu') if _force_cpu else torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"[HDC] Using device: {DEVICE}")


class Hypervector(ABC):
    """Abstract base class for all Hypervector representations."""

    def __init__(self, data):
        if isinstance(data, torch.Tensor):
            self.data = data.to(DEVICE)
        else:
            self.data = torch.tensor(data, device=DEVICE)
        self.dim = self.data.shape[0]

    @abstractmethod
    def bind(self, other):
        pass

    @abstractmethod
    def bundle(self, others):
        pass

    @classmethod
    @abstractmethod
    def batch_bundle(cls, stacked_data):
        """Bundle a batch of raw tensors (N, D) into a single HV."""
        pass

    @abstractmethod
    def permute(self, shift=1):
        pass

    @abstractmethod
    def similarity(self, other):
        pass

    @classmethod
    @abstractmethod
    def random(cls, dim):
        pass


class BinaryHypervector(Hypervector):
    """Binary Hypervector {0, 1}^D — BSC (Binary Spatter Codes)."""

    def __init__(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to(torch.int8)
        else:
            data = torch.tensor(data, dtype=torch.int8)
        super().__init__(data)

    def bind(self, other):
        return BinaryHypervector(torch.bitwise_xor(self.data, other.data))

    def bundle(self, others):
        all_data = torch.stack([self.data] + [o.data for o in others])
        return BinaryHypervector.batch_bundle(all_data)

    @classmethod
    def batch_bundle(cls, stacked_data):
        n = stacked_data.shape[0]
        if n == 1:
            return cls(stacked_data[0].clone())
        summed = stacked_data.to(torch.int32).sum(dim=0)
        threshold = n / 2.0
        result = (summed > threshold).to(torch.int8)
        ties = (summed.float() == threshold)
        if ties.any():
            result[ties] = torch.randint(
                0, 2, (ties.sum().item(),), device=stacked_data.device, dtype=torch.int8
            )
        return cls(result)

    def permute(self, shift=1):
        return BinaryHypervector(torch.roll(self.data, shift))

    def similarity(self, other):
        xor_res = torch.bitwise_xor(self.data, other.data)
        hamming_dist = xor_res.sum().item()
        return 1.0 - (hamming_dist / self.dim)

    @classmethod
    def random(cls, dim):
        return cls(torch.randint(0, 2, (dim,), device=DEVICE, dtype=torch.int8))


class FloatHypervector(Hypervector):
    """Real-valued Hypervector R^D — Normalized Gaussian."""

    def __init__(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to(torch.float32)
        else:
            data = torch.tensor(data, dtype=torch.float32)
        super().__init__(data)

    def bind(self, other):
        return FloatHypervector(self.data * other.data)

    def bundle(self, others):
        all_data = torch.stack([self.data] + [o.data for o in others])
        return FloatHypervector.batch_bundle(all_data)

    @classmethod
    def batch_bundle(cls, stacked_data):
        n = stacked_data.shape[0]
        if n == 1:
            return cls(stacked_data[0].clone())
        summed = stacked_data.sum(dim=0)
        norm = torch.linalg.norm(summed)
        if norm > 0:
            summed = summed / norm
        return cls(summed)

    def permute(self, shift=1):
        return FloatHypervector(torch.roll(self.data, shift))

    def similarity(self, other):
        dot = torch.dot(self.data, other.data)
        norm_a = torch.linalg.norm(self.data)
        norm_b = torch.linalg.norm(other.data)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return (dot / (norm_a * norm_b)).item()

    @classmethod
    def random(cls, dim):
        vec = torch.randn(dim, device=DEVICE, dtype=torch.float32)
        vec = vec / torch.linalg.norm(vec)
        return cls(vec)


class BipolarHypervector(Hypervector):
    """Bipolar Hypervector {-1, 1}^D — MAP (Multiply-Add-Permute)."""

    def __init__(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to(torch.int8)
        else:
            data = torch.tensor(data, dtype=torch.int8)
        super().__init__(data)

    def bind(self, other):
        return BipolarHypervector(self.data * other.data)

    def bundle(self, others):
        all_data = torch.stack([self.data] + [o.data for o in others])
        return BipolarHypervector.batch_bundle(all_data)

    @classmethod
    def batch_bundle(cls, stacked_data):
        n = stacked_data.shape[0]
        if n == 1:
            return cls(stacked_data[0].clone())
        summed = stacked_data.to(torch.int32).sum(dim=0)
        result = torch.where(
            summed > 0,
            torch.tensor(1, dtype=torch.int8, device=stacked_data.device),
            torch.tensor(-1, dtype=torch.int8, device=stacked_data.device)
        )
        ties = (summed == 0)
        if ties.any():
            result[ties] = (
                torch.randint(0, 2, (ties.sum().item(),),
                              device=stacked_data.device, dtype=torch.int8) * 2 - 1
            )
        return cls(result)

    def permute(self, shift=1):
        return BipolarHypervector(torch.roll(self.data, shift))

    def similarity(self, other):
        dot = torch.dot(self.data.float(), other.data.float())
        return (dot / self.dim).item()

    @classmethod
    def random(cls, dim):
        return cls(torch.randint(0, 2, (dim,), device=DEVICE, dtype=torch.int8) * 2 - 1)
