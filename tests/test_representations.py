import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import BinaryHypervector, FloatHypervector

def test_binary_hv():
    print("Testing BinaryHypervector...")
    dim = 10000
    hv1 = BinaryHypervector.random(dim)
    hv2 = BinaryHypervector.random(dim)
    
    # Test Binding (XOR)
    bound = hv1.bind(hv2)
    # Binary binding is its own inverse: (A * B) * B = A
    inverse = bound.bind(hv2)
    assert np.array_equal(hv1.data, inverse.data), "Binary Binding Inverse failed"
    print("  Binary Binding Inverse: passed")
    
    # Test Similarity
    sim_self = hv1.similarity(hv1)
    assert sim_self == 1.0, f"Binary Self-similarity failed: {sim_self}"
    
    sim_random = hv1.similarity(hv2)
    print(f"  Binary Random Similarity: {sim_random:.4f} (expected ~0.5)")
    assert 0.45 < sim_random < 0.55, f"Binary Random Similarity out of bounds: {sim_random}"
    
    # Test Bundling
    bundled = hv1.bundle([hv2])
    sim_with_1 = bundled.similarity(hv1)
    sim_with_2 = bundled.similarity(hv2)
    print(f"  Binary Bundling Similarity: hv1={sim_with_1:.4f}, hv2={sim_with_2:.4f} (expected > 0.5)")
    assert sim_with_1 > 0.50 and sim_with_2 > 0.50
    print("Testing BinaryHypervector: ALL PASSED")

def test_float_hv():
    print("\nTesting FloatHypervector...")
    dim = 10000
    hv1 = FloatHypervector.random(dim)
    hv2 = FloatHypervector.random(dim)
    
    # Test Binding (Multiplication)
    bound = hv1.bind(hv2)
    # Float binding properties: bound should be orthogonal to constituents
    sim_b1 = bound.similarity(hv1)
    sim_b2 = bound.similarity(hv2)
    print(f"  Float bound similarity to constituents: {sim_b1:.4f}, {sim_b2:.4f} (expected ~0.0)")
    assert abs(sim_b1) < 0.05 and abs(sim_b2) < 0.05
    
    # Test Similarity
    sim_self = hv1.similarity(hv1)
    assert abs(sim_self - 1.0) < 1e-6, f"Float Self-similarity failed: {sim_self}"
    
    sim_random = hv1.similarity(hv2)
    print(f"  Float Random Similarity: {sim_random:.4f} (expected ~0.0)")
    assert abs(sim_random) < 0.05, f"Float Random Similarity out of bounds: {sim_random}"
    
    # Test Bundling
    bundled = hv1.bundle([hv2])
    sim_with_1 = bundled.similarity(hv1)
    sim_with_2 = bundled.similarity(hv2)
    print(f"  Float Bundling Similarity: hv1={sim_with_1:.4f}, hv2={sim_with_2:.4f} (expected ~0.707)")
    # For two orthogonal vectors A, B: (A+B)/|A+B| similarity to A is 1/sqrt(2) approx 0.707
    assert 0.65 < sim_with_1 < 0.75 and 0.65 < sim_with_2 < 0.75
    print("Testing FloatHypervector: ALL PASSED")

if __name__ == "__main__":
    try:
        test_binary_hv()
        test_float_hv()
        print("\nAll Hypervector representation tests passed!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
