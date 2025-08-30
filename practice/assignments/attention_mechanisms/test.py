"""
Test suite for attention mechanisms assignment.
"""

import numpy as np
import sys
from pathlib import Path

# Add the assignment directory to path
sys.path.append(str(Path(__file__).parent))

from skeleton import *
from reference import *

def test_function(name, student_func, reference_func, test_cases, tolerance=1e-4, check_weights=False):
    """Test a single attention function."""
    print(f"\nTesting {name}...")
    
    all_passed = True
    
    for i, (args, kwargs) in enumerate(test_cases):
        try:
            student_result = student_func(*args, **kwargs)
            reference_result = reference_func(*args, **kwargs)
            
            # Handle functions that might not be implemented
            if student_result is None or (isinstance(student_result, tuple) and student_result[0] is None):
                print(f"  ❌ Test {i+1}: Function not implemented")
                all_passed = False
                continue
            
            # Handle both single outputs and (output, weights) tuples
            if isinstance(student_result, tuple) and isinstance(reference_result, tuple):
                student_out, student_weights = student_result
                reference_out, reference_weights = reference_result
                
                # Check output
                output_error = np.max(np.abs(student_out - reference_out))
                if output_error > tolerance:
                    print(f"  ❌ Test {i+1}: output error = {output_error:.2e} (tolerance = {tolerance:.2e})")
                    all_passed = False
                    continue
                
                # Check attention weights if requested
                if check_weights and student_weights is not None and reference_weights is not None:
                    weights_error = np.max(np.abs(student_weights - reference_weights))
                    if weights_error > tolerance:
                        print(f"  ❌ Test {i+1}: weights error = {weights_error:.2e}")
                        all_passed = False
                        continue
                    print(f"  ✅ Test {i+1}: output_err = {output_error:.2e}, weights_err = {weights_error:.2e}")
                else:
                    print(f"  ✅ Test {i+1}: output_err = {output_error:.2e}")
            else:
                # Single output comparison
                error = np.max(np.abs(student_result - reference_result))
                if error > tolerance:
                    print(f"  ❌ Test {i+1}: error = {error:.2e} (tolerance = {tolerance:.2e})")
                    all_passed = False
                    continue
                print(f"  ✅ Test {i+1}: error = {error:.2e}")
            
        except Exception as e:
            print(f"  ❌ Test {i+1} failed with exception: {e}")
            all_passed = False
    
    return all_passed

def test_class_based_attention(name, student_class, reference_class, test_configs, tolerance=1e-4):
    """Test class-based attention implementations."""
    print(f"\nTesting {name}...")
    
    all_passed = True
    np.random.seed(42)  # For reproducible weight initialization
    
    for i, config in enumerate(test_configs):
        try:
            # Initialize both implementations
            student_attn = student_class(**config)
            reference_attn = reference_class(**config)
            
            # Copy weights from student to reference for fair comparison
            if hasattr(student_attn, 'W_q'):
                reference_attn.W_q.weight.data = torch.tensor(student_attn.W_q.T)
                reference_attn.W_k.weight.data = torch.tensor(student_attn.W_k.T)
                reference_attn.W_v.weight.data = torch.tensor(student_attn.W_v.T)
                reference_attn.W_o.weight.data = torch.tensor(student_attn.W_o.T)
            
            # Generate test data
            batch_size, seq_len, d_model = 2, 8, config.get('d_model', 64)
            
            query = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            key = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            value = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            
            # Forward pass
            student_result = student_attn.forward(query, key, value, training=False)
            reference_result = reference_attn(
                torch.tensor(query), torch.tensor(key), torch.tensor(value), training=False
            )
            
            if student_result is None or (isinstance(student_result, tuple) and student_result[0] is None):
                print(f"  ❌ Test {i+1}: Forward pass not implemented")
                all_passed = False
                continue
            
            student_out = student_result[0] if isinstance(student_result, tuple) else student_result
            reference_out = reference_result[0] if isinstance(reference_result, tuple) else reference_result
            
            error = np.max(np.abs(student_out - reference_out))
            if error > tolerance:
                print(f"  ❌ Test {i+1}: error = {error:.2e} (tolerance = {tolerance:.2e})")
                all_passed = False
                continue
            
            print(f"  ✅ Test {i+1}: error = {error:.2e}")
            
        except Exception as e:
            print(f"  ❌ Test {i+1} failed with exception: {e}")
            all_passed = False
    
    return all_passed

def run_tests():
    """Run all attention mechanism tests."""
    print("Running Attention Mechanisms Tests")
    print("=" * 60)
    
    # Set seeds for reproducible tests
    np.random.seed(42)
    
    # Test data setup
    batch_size, seq_len_q, seq_len_k, d_k, d_v = 2, 8, 10, 16, 16
    
    # Test cases for SDPA
    sdpa_tests = []
    for i in range(3):
        query = np.random.randn(batch_size, seq_len_q, d_k).astype(np.float32)
        key = np.random.randn(batch_size, seq_len_k, d_k).astype(np.float32)
        value = np.random.randn(batch_size, seq_len_k, d_v).astype(np.float32)
        sdpa_tests.append(((query, key, value), {"training": False}))
    
    # Test with causal mask
    query = np.random.randn(batch_size, seq_len_q, d_k).astype(np.float32)
    key = np.random.randn(batch_size, seq_len_q, d_k).astype(np.float32)  # Same length for causal
    value = np.random.randn(batch_size, seq_len_q, d_v).astype(np.float32)
    mask = np.tril(np.ones((seq_len_q, seq_len_q)), dtype=bool)
    sdpa_tests.append(((query, key, value, mask), {"training": False}))
    
    # Utility function tests
    causal_tests = [
        ((4,), {}),
        ((8,), {}),
        ((16,), {}),
    ]
    
    padding_tests = [
        (([3, 5, 4], 6), {}),
        (([2, 8, 6, 7], 8), {}),
    ]
    
    # Class-based attention configs
    mha_configs = [
        {"d_model": 64, "num_heads": 4},
        {"d_model": 128, "num_heads": 8},
        {"d_model": 32, "num_heads": 2},
    ]
    
    mqa_configs = [
        {"d_model": 64, "num_heads": 4},
        {"d_model": 128, "num_heads": 8},
    ]
    
    gqa_configs = [
        {"d_model": 64, "num_query_heads": 8, "num_kv_heads": 2},
        {"d_model": 128, "num_query_heads": 16, "num_kv_heads": 4},
    ]
    
    mla_configs = [
        {"d_model": 64, "num_heads": 4, "latent_dim": 16},
        {"d_model": 128, "num_heads": 8, "latent_dim": 32},
    ]
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Function-based tests
    function_tests = [
        ("SDPA", scaled_dot_product_attention, AttentionReference.scaled_dot_product_attention, 
         sdpa_tests, 1e-4, True),
        ("Causal Mask", create_causal_mask, UtilityFunctions.create_causal_mask, 
         causal_tests, 1e-6, False),
        ("Padding Mask", create_padding_mask, UtilityFunctions.create_padding_mask, 
         padding_tests, 1e-6, False),
    ]
    
    for test_info in function_tests:
        name, student_func, reference_func, test_cases, tol, check_weights = test_info
        total_tests += 1
        if test_function(name, student_func, reference_func, test_cases, tol, check_weights):
            tests_passed += 1
    
    # Class-based tests
    class_tests = [
        ("Multi-Head Attention", MultiHeadAttention, MultiHeadAttentionReference, mha_configs),
        ("Multi-Query Attention", MultiQueryAttention, MultiQueryAttentionReference, mqa_configs),
        ("Grouped-Query Attention", GroupedQueryAttention, GroupedQueryAttentionReference, gqa_configs),
        ("Multi-head Latent Attention", MultiHeadLatentAttention, MultiHeadLatentAttentionReference, mla_configs),
    ]
    
    for name, student_class, reference_class, configs in class_tests:
        total_tests += 1
        if test_class_based_attention(name, student_class, reference_class, configs):
            tests_passed += 1
    
    # Test softmax separately
    print(f"\nTesting Softmax...")
    try:
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        student_result = softmax(x)
        reference_result = torch.nn.functional.softmax(torch.tensor(x), dim=-1).numpy()
        
        if student_result is not None:
            error = np.max(np.abs(student_result - reference_result))
            if error < 1e-6:
                print(f"  ✅ Softmax: error = {error:.2e}")
                tests_passed += 1
            else:
                print(f"  ❌ Softmax: error = {error:.2e}")
        else:
            print(f"  ❌ Softmax: not implemented")
    except Exception as e:
        print(f"  ❌ Softmax failed: {e}")
    
    total_tests += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {tests_passed}/{total_tests} tests passed")
    print(f"{'='*60}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Great work!")
        print("\n📝 Key concepts you've implemented:")
        print("   • Scaled Dot-Product Attention (SDPA)")
        print("   • Multi-Head Attention (MHA)")
        print("   • Multi-Query Attention (MQA) - efficient KV sharing")
        print("   • Grouped-Query Attention (GQA) - balanced efficiency")
        print("   • Multi-head Latent Attention (MLA) - memory-efficient compression")
        print("   • Attention masking (causal and padding)")
    else:
        print("❌ Some tests failed. Check your implementation and try again.")
        print("\n💡 Debug tips:")
        print("   • Start with SDPA - it's the foundation for all others")
        print("   • Check tensor shapes carefully - broadcasting is key")
        print("   • Verify softmax numerical stability")
        print("   • MQA/GQA/MLA require careful weight matrix management")

if __name__ == "__main__":
    run_tests()