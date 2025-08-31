#!/usr/bin/env python3
"""
Unified test runner for all deep learning practice assignments.
Usage: python test_runner.py <assignment_name> or python test_runner.py all
"""

import torch
import torch.nn.functional as F
import sys
import traceback
import importlib
from pathlib import Path
from typing import Any, Callable, List, Tuple, Dict, Optional

class TestRunner:
    """Unified test runner with fail-fast behavior and full stack traces."""
    
    def __init__(self, stop_on_failure=True):
        self.stop_on_failure = stop_on_failure
        self.tests_passed = 0
        self.total_tests = 0
        
    def test_function(self, name: str, student_func: Callable, reference_func: Callable, 
                     test_cases: List[Tuple], tolerance: float = 1e-5, 
                     check_weights: bool = False, has_backward: bool = False) -> bool:
        """Generic test function that handles different assignment types."""
        print(f"\nTesting {name}...")
        
        for i, test_case in enumerate(test_cases):
            try:
                # Handle different test case formats
                if isinstance(test_case, tuple) and len(test_case) == 2:
                    args, kwargs = test_case
                    if isinstance(args, tuple):
                        student_result = student_func(*args, **kwargs)
                        reference_result = reference_func(*args, **kwargs)
                    else:
                        # Single argument case
                        student_result = student_func(args, **kwargs)
                        reference_result = reference_func(args, **kwargs)
                else:
                    # Direct argument case (for activation functions)
                    student_result = student_func(test_case)
                    reference_result = reference_func(test_case)
                
                # Handle functions that might not be implemented
                if student_result is None or (isinstance(student_result, tuple) and student_result[0] is None):
                    print(f"  ❌ Test {i+1}: Function not implemented")
                    if self.stop_on_failure:
                        return False
                    continue
                
                # Handle different return types
                if has_backward and isinstance(student_result, tuple) and len(student_result) == 2:
                    # Activation functions with forward/backward
                    student_out, cache = student_result
                    reference_out = reference_result
                    
                    # Test forward pass
                    forward_error = torch.max(torch.abs(student_out - reference_out)).item()
                    if forward_error > tolerance:
                        print(f"  ❌ Test {i+1}: forward error = {forward_error:.2e} (tolerance = {tolerance:.2e})")
                        if self.stop_on_failure:
                            return False
                        continue
                        
                    print(f"  ✅ Test {i+1}: forward_err = {forward_error:.2e}")
                    
                elif isinstance(student_result, tuple) and isinstance(reference_result, tuple):
                    # Handle functions that return multiple values (like attention or RoPE)
                    if len(student_result) != len(reference_result):
                        print(f"  ❌ Test {i+1}: Output length mismatch")
                        if self.stop_on_failure:
                            return False
                        continue
                    
                    if len(student_result) == 2 and check_weights:
                        # Attention with weights
                        student_out, student_weights = student_result
                        reference_out, reference_weights = reference_result
                        
                        output_error = torch.max(torch.abs(student_out - reference_out)).item()
                        if output_error > tolerance:
                            print(f"  ❌ Test {i+1}: output error = {output_error:.2e}")
                            if self.stop_on_failure:
                                return False
                            continue
                            
                        if student_weights is not None and reference_weights is not None:
                            weights_error = torch.max(torch.abs(student_weights - reference_weights)).item()
                            if weights_error > tolerance:
                                print(f"  ❌ Test {i+1}: weights error = {weights_error:.2e}")
                                if self.stop_on_failure:
                                    return False
                                continue
                            print(f"  ✅ Test {i+1}: out_err = {output_error:.2e}, weights_err = {weights_error:.2e}")
                        else:
                            print(f"  ✅ Test {i+1}: output_err = {output_error:.2e}")
                    else:
                        # Multiple outputs (like RoPE cos/sin)
                        max_error = 0
                        for j, (s_out, r_out) in enumerate(zip(student_result, reference_result)):
                            error = torch.max(torch.abs(s_out - r_out)).item()
                            max_error = max(max_error, error)
                            
                        if max_error > tolerance:
                            print(f"  ❌ Test {i+1}: error = {max_error:.2e}")
                            if self.stop_on_failure:
                                return False
                            continue
                        print(f"  ✅ Test {i+1}: error = {max_error:.2e}")
                        
                else:
                    # Single output comparison
                    if student_result.shape != reference_result.shape:
                        print(f"  ❌ Test {i+1}: Shape mismatch. Expected {reference_result.shape}, got {student_result.shape}")
                        if self.stop_on_failure:
                            return False
                        continue
                        
                    error = torch.max(torch.abs(student_result - reference_result)).item()
                    if error > tolerance:
                        print(f"  ❌ Test {i+1}: error = {error:.2e} (tolerance = {tolerance:.2e})")
                        if self.stop_on_failure:
                            return False
                        continue
                    print(f"  ✅ Test {i+1}: error = {error:.2e}")
                
            except Exception as e:
                print(f"  ❌ Test {i+1} failed with exception: {e}")
                print("Full traceback:")
                traceback.print_exc()
                if self.stop_on_failure:
                    return False
        
        return True
    
    def test_class_based(self, name: str, student_class: type, reference_class: type, 
                        test_configs: List[Dict], tolerance: float = 1e-4) -> bool:
        """Test class-based implementations like attention mechanisms."""
        print(f"\nTesting {name}...")
        
        torch.manual_seed(42)  # Reproducible initialization
        
        for i, config in enumerate(test_configs):
            try:
                # Initialize both implementations
                student_instance = student_class(**config)
                reference_instance = reference_class(**config)
                
                # Copy weights for fair comparison
                if hasattr(student_instance, 'W_q') and hasattr(reference_instance, 'W_q'):
                    reference_instance.W_q.weight.data.copy_(student_instance.W_q.weight.data)
                    
                    # Handle different attention mechanism weight structures
                    if hasattr(student_instance, 'W_k') and hasattr(reference_instance, 'W_k'):
                        reference_instance.W_k.weight.data.copy_(student_instance.W_k.weight.data)
                    if hasattr(student_instance, 'W_v') and hasattr(reference_instance, 'W_v'):
                        reference_instance.W_v.weight.data.copy_(student_instance.W_v.weight.data)
                    
                    # Handle MLA-specific weights
                    if hasattr(student_instance, 'W_kc') and hasattr(reference_instance, 'W_kc'):
                        reference_instance.W_kc.weight.data.copy_(student_instance.W_kc.weight.data)
                        reference_instance.W_vc.weight.data.copy_(student_instance.W_vc.weight.data)
                        reference_instance.W_ku.weight.data.copy_(student_instance.W_ku.weight.data)
                        reference_instance.W_vu.weight.data.copy_(student_instance.W_vu.weight.data)
                    
                    if hasattr(student_instance, 'W_o') and hasattr(reference_instance, 'W_o'):
                        reference_instance.W_o.weight.data.copy_(student_instance.W_o.weight.data)
                
                # Generate test data
                batch_size, seq_len, d_model = 2, 8, config.get('d_model', 64)
                query = torch.randn(batch_size, seq_len, d_model)
                key = torch.randn(batch_size, seq_len, d_model)
                value = torch.randn(batch_size, seq_len, d_model)
                
                # Test forward pass
                student_result = student_instance.forward(query, key, value, training=False)
                reference_result = reference_instance(query, key, value, training=False)
                
                if student_result is None or (isinstance(student_result, tuple) and student_result[0] is None):
                    print(f"  ❌ Test {i+1}: Forward pass not implemented")
                    if self.stop_on_failure:
                        return False
                    continue
                
                student_out = student_result[0] if isinstance(student_result, tuple) else student_result
                reference_out = reference_result[0] if isinstance(reference_result, tuple) else reference_result
                
                # Ensure both outputs are torch tensors
                if not isinstance(student_out, torch.Tensor):
                    student_out = torch.tensor(student_out)
                if not isinstance(reference_out, torch.Tensor):
                    reference_out = torch.tensor(reference_out)
                
                # Detach tensors that require gradients
                if hasattr(student_out, 'requires_grad') and student_out.requires_grad:
                    student_out = student_out.detach()
                if hasattr(reference_out, 'requires_grad') and reference_out.requires_grad:
                    reference_out = reference_out.detach()
                
                error = torch.max(torch.abs(student_out - reference_out)).item()
                if error > tolerance:
                    print(f"  ❌ Test {i+1}: error = {error:.2e}")
                    if self.stop_on_failure:
                        return False
                    continue
                
                print(f"  ✅ Test {i+1}: error = {error:.2e}")
                
            except Exception as e:
                print(f"  ❌ Test {i+1} failed with exception: {e}")
                print("Full traceback:")
                traceback.print_exc()
                if self.stop_on_failure:
                    return False
        
        return True
    
    def run_assignment_tests(self, assignment_name: str, test_filter: str = None) -> bool:
        """Run tests for a specific assignment, optionally filtering by test name."""
        assignment_dir = Path(f"assignments/{assignment_name}")
        if not assignment_dir.exists():
            print(f"❌ Assignment '{assignment_name}' not found!")
            return False
        
        if test_filter:
            print(f"\n{'='*60}")
            print(f"Testing: {assignment_name.replace('_', ' ').title()} - {test_filter}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Testing Assignment: {assignment_name.replace('_', ' ').title()}")
            print(f"{'='*60}")
        
        # Add assignment directory to path
        sys.path.insert(0, str(assignment_dir))
        
        try:
            # Import modules
            skeleton = importlib.import_module("skeleton")
            reference = importlib.import_module("reference")
            
            # Set random seeds
            torch.manual_seed(42)
            
            assignment_passed = True
            
            if assignment_name == "activation_functions":
                assignment_passed = self._test_activation_functions(skeleton, reference, test_filter)
            elif assignment_name == "positional_embeddings":
                assignment_passed = self._test_positional_embeddings(skeleton, reference, test_filter)
            elif assignment_name == "attention_mechanisms":
                assignment_passed = self._test_attention_mechanisms(skeleton, reference, test_filter)
            else:
                print(f"❌ Unknown assignment: {assignment_name}")
                return False
            
            return assignment_passed
            
        except Exception as e:
            print(f"❌ Failed to load assignment '{assignment_name}': {e}")
            print("Full traceback:")
            traceback.print_exc()
            return False
        finally:
            # Clean up path
            if str(assignment_dir) in sys.path:
                sys.path.remove(str(assignment_dir))
    
    def _test_activation_functions(self, skeleton, reference, test_filter=None) -> bool:
        """Test activation functions assignment."""
        # Test inputs
        test_inputs = [
            torch.tensor([1.0, -1.0, 0.0, 2.0, -2.0]),
            torch.tensor([[1.0, -0.5], [0.0, 2.0]]),
            torch.randn(3, 4, 5) * 2,
            torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0]),
        ]
        
        tests = [
            ("ReLU", skeleton.relu_forward, reference.ActivationFunctions.relu, {}),
            ("Leaky ReLU", skeleton.leaky_relu_forward, reference.ActivationFunctions.leaky_relu, {}),
            ("Sigmoid", skeleton.sigmoid_forward, reference.ActivationFunctions.sigmoid, {}),
            ("Tanh", skeleton.tanh_forward, reference.ActivationFunctions.tanh, {}),
            ("Swish", skeleton.swish_forward, reference.ActivationFunctions.swish, {}),
            ("GELU (exact)", skeleton.gelu_forward, reference.ActivationFunctions.gelu, {}),
            ("GELU (approx)", skeleton.gelu_forward, reference.ActivationFunctions.gelu, {"approximate": True}),
            ("ELU", skeleton.elu_forward, reference.ActivationFunctions.elu, {}),
        ]
        
        for name, student_func, reference_func, kwargs in tests:
            # Skip tests that don't match filter
            if test_filter and test_filter.lower() not in name.lower():
                continue
                
            self.total_tests += 1
            test_cases = [(inp, kwargs) for inp in test_inputs]
            if not self.test_function(name, student_func, reference_func, test_cases, 
                                    tolerance=1e-5, has_backward=True):
                if self.stop_on_failure:
                    print(f"\n🛑 STOPPING: {name} test failed")
                    print(f"Results: {self.tests_passed}/{self.total_tests} tests passed before failure")
                    return False
            else:
                self.tests_passed += 1
        
        if test_filter and self.total_tests == 0:
            print(f"❌ No tests found matching '{test_filter}'")
            print("Available tests: ReLU, Leaky ReLU, Sigmoid, Tanh, Swish, GELU, ELU")
            return False
            
        return True
    
    def _test_positional_embeddings(self, skeleton, reference, test_filter=None) -> bool:
        """Test positional embeddings assignment."""
        tests = [
            ("Sinusoidal PE", skeleton.sinusoidal_positional_encoding, 
             reference.PositionalEmbeddings.sinusoidal_positional_encoding,
             [((8, 4), {}), ((16, 8), {}), ((10, 6), {"base": 1000})]),
            ("Learnable PE", skeleton.learnable_positional_embedding,
             reference.PositionalEmbeddings.learnable_positional_embedding,
             [((8, 4), {}), ((16, 8), {}), ((10, 6), {"init_std": 0.01})]),
            ("RoPE Encoding", skeleton.rope_positional_encoding,
             reference.PositionalEmbeddings.rope_positional_encoding,
             [((8, 4), {}), ((16, 8), {}), ((10, 6), {"base": 1000})]),
            ("Relative Position", skeleton.relative_position_encoding,
             reference.PositionalEmbeddings.relative_position_encoding,
             [((8,), {}), ((16,), {}), ((10,), {"max_distance": 16})]),
            ("ALiBi Slopes", skeleton.alibi_slopes,
             reference.PositionalEmbeddings.alibi_slopes,
             [((4,), {}), ((8,), {}), ((6,), {})]),
        ]
        
        for name, student_func, reference_func, test_cases in tests:
            # Skip tests that don't match filter
            if test_filter and test_filter.lower() not in name.lower():
                continue
                
            self.total_tests += 1
            if not self.test_function(name, student_func, reference_func, test_cases):
                if self.stop_on_failure:
                    print(f"\n🛑 STOPPING: {name} test failed")
                    print(f"Results: {self.tests_passed}/{self.total_tests} tests passed before failure")
                    return False
            else:
                self.tests_passed += 1
        
        if test_filter and self.total_tests == 0:
            print(f"❌ No tests found matching '{test_filter}'")
            print("Available tests: Sinusoidal PE, Learnable PE, RoPE Encoding, Relative Position, ALiBi Slopes")
            return False
            
        return True
    
    def _test_attention_mechanisms(self, skeleton, reference, test_filter=None) -> bool:
        """Test attention mechanisms assignment."""
        # Test SDPA with self-attention (same seq_len for Q, K, V)
        batch_size, seq_len, d_k, d_v = 2, 8, 16, 16
        sdpa_tests = []
        for i in range(3):
            query = torch.randn(batch_size, seq_len, d_k)
            key = torch.randn(batch_size, seq_len, d_k)
            value = torch.randn(batch_size, seq_len, d_v)
            sdpa_tests.append(((query, key, value), {"training": False}))
        
        # Add one cross-attention test case with different seq lengths
        query_cross = torch.randn(batch_size, 6, d_k)  # Different query length
        key_cross = torch.randn(batch_size, seq_len, d_k)
        value_cross = torch.randn(batch_size, seq_len, d_v)
        sdpa_tests.append(((query_cross, key_cross, value_cross), {"training": False}))
        
        # Test utility functions
        causal_tests = [((4,), {}), ((8,), {})]
        padding_tests = [(([3, 5, 4], 6), {}), (([2, 8, 6, 7], 8), {})]
        
        function_tests = [
            ("SDPA", skeleton.scaled_dot_product_attention, 
             reference.AttentionReference.scaled_dot_product_attention, 
             sdpa_tests, 1e-4, True),
            ("Causal Mask", skeleton.create_causal_mask,
             reference.UtilityFunctions.create_causal_mask,
             causal_tests, 1e-6, False),
            ("Padding Mask", skeleton.create_padding_mask,
             reference.UtilityFunctions.create_padding_mask,
             padding_tests, 1e-6, False),
        ]
        
        for name, student_func, reference_func, test_cases, tol, check_weights in function_tests:
            # Skip tests that don't match filter
            if test_filter and test_filter.lower() not in name.lower():
                continue
                
            self.total_tests += 1
            if not self.test_function(name, student_func, reference_func, test_cases, 
                                    tolerance=tol, check_weights=check_weights):
                if self.stop_on_failure:
                    print(f"\n🛑 STOPPING: {name} test failed")
                    print(f"Results: {self.tests_passed}/{self.total_tests} tests passed before failure")
                    return False
            else:
                self.tests_passed += 1
        
        # Test class-based attention mechanisms
        mha_configs = [
            {"d_model": 64, "num_heads": 4},
            {"d_model": 32, "num_heads": 2},
        ]
        
        mqa_configs = [
            {"d_model": 64, "num_heads": 4},
        ]
        
        gqa_configs = [
            {"d_model": 64, "num_query_heads": 8, "num_kv_heads": 2},
        ]
        
        mla_configs = [
            {"d_model": 64, "num_heads": 4, "latent_dim": 16},
        ]
        
        class_tests = [
            ("Multi-Head Attention", skeleton.MultiHeadAttention, reference.MultiHeadAttentionReference, mha_configs),
            ("Multi-Query Attention", skeleton.MultiQueryAttention, reference.MultiQueryAttentionReference, mqa_configs),  
            ("Grouped-Query Attention", skeleton.GroupedQueryAttention, reference.GroupedQueryAttentionReference, gqa_configs),
            ("Multi-head Latent Attention", skeleton.MultiHeadLatentAttention, reference.MultiHeadLatentAttentionReference, mla_configs),
        ]
        
        for name, student_class, reference_class, configs in class_tests:
            # Skip tests that don't match filter
            if test_filter and test_filter.lower() not in name.lower():
                continue
                
            self.total_tests += 1
            try:
                if self.test_class_based(name, student_class, reference_class, configs):
                    self.tests_passed += 1
                else:
                    if self.stop_on_failure:
                        print(f"\n🛑 STOPPING: {name} test failed")
                        print(f"Results: {self.tests_passed}/{self.total_tests} tests passed before failure")
                        return False
            except Exception as e:
                print(f"\n❌ {name} test setup failed: {e}")
                print("Full traceback:")
                traceback.print_exc()
                if self.stop_on_failure:
                    print(f"\n🛑 STOPPING: {name} test setup failed")
                    print(f"Results: {self.tests_passed}/{self.total_tests} tests passed before failure")
                    return False
        
        # Test softmax function
        if not test_filter or "softmax" in test_filter.lower():
            print(f"\nTesting Softmax...")
            try:
                x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                student_result = skeleton.softmax(x)
                reference_result = torch.nn.functional.softmax(x, dim=-1)
                
                self.total_tests += 1
                if student_result is not None:
                    error = torch.max(torch.abs(student_result - reference_result)).item()
                    if error < 1e-6:
                        print(f"  ✅ Softmax: error = {error:.2e}")
                        self.tests_passed += 1
                    else:
                        print(f"  ❌ Softmax: error = {error:.2e}")
                        if self.stop_on_failure:
                            print(f"\n🛑 STOPPING: Softmax test failed")
                            return False
                else:
                    print(f"  ❌ Softmax: not implemented")
                    if self.stop_on_failure:
                        print(f"\n🛑 STOPPING: Softmax not implemented") 
                        return False
            except Exception as e:
                print(f"  ❌ Softmax failed: {e}")
                print("Full traceback:")
                traceback.print_exc()
                if self.stop_on_failure:
                    print(f"\n🛑 STOPPING: Softmax test failed")
                    return False
        
        # Check if filter matched anything
        if test_filter and self.total_tests == 0:
            print(f"❌ No tests found matching '{test_filter}'")
            print("Available tests: SDPA, Causal Mask, Padding Mask, Multi-Head Attention, Multi-Query Attention, Grouped-Query Attention, Multi-head Latent Attention, Softmax")
            return False
        
        return True

def main():
    """Main entry point."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python test_runner.py <assignment_name|all> [test_filter]")
        print("Available assignments: activation_functions, positional_embeddings, attention_mechanisms")
        print("\nExamples:")
        print("  python test_runner.py attention_mechanisms")
        print("  python test_runner.py attention_mechanisms SDPA")
        print("  python test_runner.py attention_mechanisms 'Multi-Head'")
        print("  python test_runner.py activation_functions ReLU")
        print("  python test_runner.py positional_embeddings RoPE")
        return
    
    assignment = sys.argv[1]
    test_filter = sys.argv[2] if len(sys.argv) == 3 else None
    runner = TestRunner(stop_on_failure=True)
    
    if assignment == "all":
        if test_filter:
            print("❌ Cannot use test filter with 'all' assignments")
            return
            
        assignments = ["activation_functions", "positional_embeddings", "attention_mechanisms"]
        total_passed = 0
        
        for assign in assignments:
            if runner.run_assignment_tests(assign):
                total_passed += 1
            else:
                break  # Stop on first failure
        
        print(f"\n{'='*60}")
        print(f"Overall Results: {total_passed}/{len(assignments)} assignments passed")
        print(f"{'='*60}")
    else:
        runner.run_assignment_tests(assignment, test_filter)

if __name__ == "__main__":
    main()