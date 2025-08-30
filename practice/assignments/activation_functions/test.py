"""
Test suite for activation functions assignment.
"""

import torch
import sys
from pathlib import Path

# Add the assignment directory to path
sys.path.append(str(Path(__file__).parent))

from skeleton import *
from reference import ActivationFunctions

def test_function(name, student_forward, student_backward, reference_forward, reference_grad, 
                  test_inputs, tolerance=1e-5, **kwargs):
    """Test a single activation function."""
    print(f"\nTesting {name}...")
    
    all_passed = True
    
    for i, x in enumerate(test_inputs):
        try:
            # Test forward pass
            student_out, cache = student_forward(x, **kwargs)
            reference_out = reference_forward(x, **kwargs)
            
            if student_out is None:
                print(f"  ❌ Forward pass not implemented")
                all_passed = False
                continue
                
            forward_error = torch.max(torch.abs(student_out - reference_out)).item()
            if forward_error > tolerance:
                print(f"  ❌ Forward pass test {i+1}: error = {forward_error:.2e} (tolerance = {tolerance:.2e})")
                all_passed = False
                continue
            
            # Test backward pass
            grad_out = torch.randn_like(student_out)
            student_grad = student_backward(grad_out, cache)
            reference_grad_val = reference_grad(x, **kwargs)
            
            if student_grad is None:
                print(f"  ❌ Backward pass not implemented")
                all_passed = False
                continue
                
            # The reference gradient is for grad_out = ones, so we need to scale
            expected_grad = grad_out * reference_grad_val
            backward_error = torch.max(torch.abs(student_grad - expected_grad)).item()
            
            if backward_error > tolerance:
                print(f"  ❌ Backward pass test {i+1}: error = {backward_error:.2e} (tolerance = {tolerance:.2e})")
                all_passed = False
                continue
                
            print(f"  ✅ Test {i+1}: forward_err = {forward_error:.2e}, backward_err = {backward_error:.2e}")
            
        except Exception as e:
            print(f"  ❌ Test {i+1} failed with exception: {e}")
            all_passed = False
    
    return all_passed

def run_tests():
    """Run all activation function tests."""
    print("Running Activation Functions Tests")
    print("=" * 50)
    
    # Test inputs
    test_inputs = [
        torch.tensor([1.0, -1.0, 0.0, 2.0, -2.0]),
        torch.tensor([[1.0, -0.5], [0.0, 2.0]]),
        torch.randn(3, 4, 5) * 2,  # Larger random tensor
        torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0]),  # Test extreme values
    ]
    
    tests = [
        ("ReLU", relu_forward, relu_backward, 
         ActivationFunctions.relu, ActivationFunctions.relu_grad),
        ("Leaky ReLU", leaky_relu_forward, leaky_relu_backward,
         ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_grad),
        ("Sigmoid", sigmoid_forward, sigmoid_backward,
         ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_grad),
        ("Tanh", tanh_forward, tanh_backward,
         ActivationFunctions.tanh, ActivationFunctions.tanh_grad),
        ("Swish", swish_forward, swish_backward,
         ActivationFunctions.swish, ActivationFunctions.swish_grad),
        ("GELU (exact)", gelu_forward, gelu_backward,
         ActivationFunctions.gelu, ActivationFunctions.gelu_grad),
        ("GELU (approx)", gelu_forward, gelu_backward,
         ActivationFunctions.gelu, ActivationFunctions.gelu_grad, {"approximate": True}),
        ("ELU", elu_forward, elu_backward,
         ActivationFunctions.elu, ActivationFunctions.elu_grad),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    # Set seed for reproducible tests
    torch.manual_seed(42)
    
    for test_info in tests:
        if len(test_info) == 6:  # Test with extra kwargs
            name, student_fwd, student_bwd, ref_fwd, ref_grad, kwargs = test_info
        else:
            name, student_fwd, student_bwd, ref_fwd, ref_grad = test_info
            kwargs = {}
        
        if test_function(name, student_fwd, student_bwd, ref_fwd, ref_grad, 
                        test_inputs, **kwargs):
            passed_tests += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*50}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Great work!")
        print("\n📝 Key concepts you've implemented:")
        print("   • ReLU and variants (Leaky ReLU, ELU)")
        print("   • Sigmoid and Tanh with numerical stability")
        print("   • Modern activations (Swish/SiLU, GELU)")
        print("   • Forward and backward pass implementations")
        print("   • PyTorch tensor operations and autograd concepts")
    else:
        print("❌ Some tests failed. Check your implementation and try again.")
        print("\n💡 Debug tips:")
        print("   • Use torch.where() for conditional operations")
        print("   • Check tensor shapes and data types")
        print("   • Be careful with numerical stability in sigmoid/tanh")
        print("   • Cache the right values for backward pass")

if __name__ == "__main__":
    run_tests()