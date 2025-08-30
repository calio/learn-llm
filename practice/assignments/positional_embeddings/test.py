"""
Test suite for positional embeddings assignment.
"""

import torch
import sys
from pathlib import Path

# Add the assignment directory to path
sys.path.append(str(Path(__file__).parent))

from skeleton import *
from reference import PositionalEmbeddings

def test_function(name, student_func, reference_func, test_cases, tolerance=1e-5):
    """Test a single positional embedding function."""
    print(f"\nTesting {name}...")
    
    all_passed = True
    
    for i, (args, kwargs) in enumerate(test_cases):
        try:
            student_out = student_func(*args, **kwargs)
            reference_out = reference_func(*args, **kwargs)
            
            if student_out is None:
                print(f"  ❌ Test {i+1}: Function not implemented")
                all_passed = False
                continue
            
            if isinstance(student_out, tuple):
                # Handle functions that return multiple values (like RoPE)
                if len(student_out) != len(reference_out):
                    print(f"  ❌ Test {i+1}: Output length mismatch")
                    all_passed = False
                    continue
                
                max_error = 0
                for j, (s_out, r_out) in enumerate(zip(student_out, reference_out)):
                    error = torch.max(torch.abs(s_out - r_out)).item()
                    max_error = max(max_error, error)
                    
                if max_error > tolerance:
                    print(f"  ❌ Test {i+1}: error = {max_error:.2e} (tolerance = {tolerance:.2e})")
                    all_passed = False
                    continue
                else:
                    print(f"  ✅ Test {i+1}: error = {max_error:.2e}")
            else:
                # Single output
                if student_out.shape != reference_out.shape:
                    print(f"  ❌ Test {i+1}: Shape mismatch. Expected {reference_out.shape}, got {student_out.shape}")
                    all_passed = False
                    continue
                
                error = torch.max(torch.abs(student_out - reference_out)).item()
                if error > tolerance:
                    print(f"  ❌ Test {i+1}: error = {error:.2e} (tolerance = {tolerance:.2e})")
                    all_passed = False
                    continue
                else:
                    print(f"  ✅ Test {i+1}: error = {error:.2e}")
            
        except Exception as e:
            print(f"  ❌ Test {i+1} failed with exception: {e}")
            all_passed = False
    
    return all_passed

def test_rope_application():
    """Test RoPE application separately since it involves two functions."""
    print(f"\nTesting RoPE Application...")
    
    test_cases = [
        (8, 4),   # Small case
        (16, 8),  # Medium case  
        (32, 16), # Larger case
    ]
    
    all_passed = True
    torch.manual_seed(42)  # For reproducible tests
    
    for i, (seq_len, d_model) in enumerate(test_cases):
        try:
            # Generate RoPE encodings
            student_cos, student_sin = rope_positional_encoding(seq_len, d_model)
            reference_cos, reference_sin = PositionalEmbeddings.rope_positional_encoding(seq_len, d_model)
            
            if student_cos is None or student_sin is None:
                print(f"  ❌ Test {i+1}: RoPE encoding not implemented")
                all_passed = False
                continue
            
            # Test random input
            x = torch.randn(2, seq_len, d_model)  # Batch dimension
            
            student_rotated = apply_rope(x, student_cos, student_sin)
            reference_rotated = PositionalEmbeddings.apply_rope(x, reference_cos, reference_sin)
            
            if student_rotated is None:
                print(f"  ❌ Test {i+1}: RoPE application not implemented")
                all_passed = False
                continue
            
            error = torch.max(torch.abs(student_rotated - reference_rotated)).item()
            if error > 1e-5:
                print(f"  ❌ Test {i+1}: error = {error:.2e}")
                all_passed = False
            else:
                print(f"  ✅ Test {i+1}: error = {error:.2e}")
                
        except Exception as e:
            print(f"  ❌ Test {i+1} failed with exception: {e}")
            all_passed = False
    
    return all_passed

def test_relative_position_pipeline():
    """Test the complete relative position pipeline."""
    print(f"\nTesting Relative Position Pipeline...")
    
    seq_len = 8
    max_distance = 4
    d_model = 6
    
    try:
        # Step 1: Generate relative positions
        student_rel_pos = relative_position_encoding(seq_len, max_distance)
        reference_rel_pos = PositionalEmbeddings.relative_position_encoding(seq_len, max_distance)
        
        if student_rel_pos is None:
            print(f"  ❌ Relative position encoding not implemented")
            return False
        
        # Step 2: Create embedding table
        student_table = create_relative_position_embeddings(max_distance, d_model)
        reference_table = PositionalEmbeddings.create_relative_position_embeddings(max_distance, d_model)
        
        if student_table is None:
            print(f"  ❌ Relative position embedding table not implemented")
            return False
        
        # Step 3: Lookup embeddings
        student_embeddings = lookup_relative_embeddings(student_rel_pos, student_table)
        reference_embeddings = PositionalEmbeddings.lookup_relative_embeddings(reference_rel_pos, reference_table)
        
        if student_embeddings is None:
            print(f"  ❌ Relative position embedding lookup not implemented")
            return False
        
        error = torch.max(torch.abs(student_embeddings - reference_embeddings)).item()
        if error > 1e-5:
            print(f"  ❌ Pipeline error = {error:.2e}")
            return False
        else:
            print(f"  ✅ Pipeline error = {error:.2e}")
            return True
            
    except Exception as e:
        print(f"  ❌ Pipeline failed with exception: {e}")
        return False

def test_alibi_pipeline():
    """Test ALiBi slope and bias generation."""
    print(f"\nTesting ALiBi Pipeline...")
    
    test_cases = [4, 8, 16, 6]  # Include non-power-of-2
    all_passed = True
    
    for i, num_heads in enumerate(test_cases):
        try:
            seq_len = 10
            
            # Generate slopes
            student_slopes = alibi_slopes(num_heads)
            reference_slopes = PositionalEmbeddings.alibi_slopes(num_heads)
            
            if student_slopes is None:
                print(f"  ❌ Test {i+1}: ALiBi slopes not implemented")
                all_passed = False
                continue
            
            # Create bias
            student_bias = create_alibi_bias(seq_len, student_slopes)
            reference_bias = PositionalEmbeddings.create_alibi_bias(seq_len, reference_slopes)
            
            if student_bias is None:
                print(f"  ❌ Test {i+1}: ALiBi bias not implemented")
                all_passed = False
                continue
            
            slopes_error = torch.max(torch.abs(student_slopes - reference_slopes)).item()
            bias_error = torch.max(torch.abs(student_bias - reference_bias)).item()
            
            if slopes_error > 1e-5 or bias_error > 1e-5:
                print(f"  ❌ Test {i+1}: slopes_err = {slopes_error:.2e}, bias_err = {bias_error:.2e}")
                all_passed = False
            else:
                print(f"  ✅ Test {i+1}: slopes_err = {slopes_error:.2e}, bias_err = {bias_error:.2e}")
                
        except Exception as e:
            print(f"  ❌ Test {i+1} failed with exception: {e}")
            all_passed = False
    
    return all_passed

def run_tests():
    """Run all positional embedding tests."""
    print("Running Positional Embeddings Tests")
    print("=" * 50)
    
    # Set seeds for reproducible tests
    torch.manual_seed(42)
    
    # Test cases: (args, kwargs)
    sinusoidal_tests = [
        ((8, 4), {}),
        ((16, 8), {}),
        ((32, 16), {}),
        ((10, 6), {"base": 1000}),  # Different base
    ]
    
    learnable_tests = [
        ((8, 4), {}),
        ((16, 8), {}),
        ((10, 6), {"init_std": 0.01}),
    ]
    
    rope_tests = [
        ((8, 4), {}),
        ((16, 8), {}),
        ((32, 16), {}),
        ((10, 6), {"base": 1000}),
    ]
    
    relative_pos_tests = [
        ((8,), {}),
        ((16,), {}),
        ((10,), {"max_distance": 16}),
    ]
    
    relative_table_tests = [
        ((4, 6), {}),
        ((8, 12), {}),
        ((16, 8), {"init_std": 0.01}),
    ]
    
    alibi_slope_tests = [
        ((4,), {}),
        ((8,), {}),
        ((6,), {}),  # Non-power-of-2
    ]
    
    tests = [
        ("Sinusoidal PE", sinusoidal_positional_encoding, 
         PositionalEmbeddings.sinusoidal_positional_encoding, sinusoidal_tests),
        ("Learnable PE", learnable_positional_embedding,
         PositionalEmbeddings.learnable_positional_embedding, learnable_tests),
        ("RoPE Encoding", rope_positional_encoding,
         PositionalEmbeddings.rope_positional_encoding, rope_tests),
        ("Relative Position", relative_position_encoding,
         PositionalEmbeddings.relative_position_encoding, relative_pos_tests),
        ("Relative Embedding Table", create_relative_position_embeddings,
         PositionalEmbeddings.create_relative_position_embeddings, relative_table_tests),
        ("ALiBi Slopes", alibi_slopes,
         PositionalEmbeddings.alibi_slopes, alibi_slope_tests),
    ]
    
    passed_tests = 0
    total_tests = len(tests) + 3  # +3 for special pipeline tests
    
    for name, student_func, reference_func, test_cases in tests:
        if test_function(name, student_func, reference_func, test_cases):
            passed_tests += 1
    
    # Special tests
    if test_rope_application():
        passed_tests += 1
    
    if test_relative_position_pipeline():
        passed_tests += 1
    
    if test_alibi_pipeline():
        passed_tests += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*50}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Great work!")
        print("\n📝 Key concepts you've implemented:")
        print("   • Sinusoidal Position Encoding (original Transformer)")
        print("   • Learnable/Absolute Position Embeddings")
        print("   • RoPE (Rotary Position Embedding) - LLaMA style")
        print("   • T5-style Relative Position Encoding")
        print("   • ALiBi (Attention with Linear Biases) - BLOOM style")
        print("   • PyTorch tensor operations and broadcasting")
    else:
        print("❌ Some tests failed. Check your implementation and try again.")
        print("\n💡 Debug tips:")
        print("   • Use torch.arange(), torch.outer(), torch.broadcast_to()")
        print("   • Check tensor shapes and data types (.long() for indices)")
        print("   • RoPE requires careful even/odd dimension handling")
        print("   • ALiBi slopes follow geometric progression patterns")

if __name__ == "__main__":
    run_tests()